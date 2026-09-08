import os

os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADCD_DIR = os.path.join(ROOT, "ADCD-Net")

if ADCD_DIR not in sys.path:
    sys.path.insert(0, ADCD_DIR)

ADCD_CKPT = os.path.join(ADCD_DIR, "ADCDNet.pth")
DOCRES_CKPT = os.path.join(ADCD_DIR, "docres.pkl")
MAX_SIZE = 512

_model = None
_device = None


def _load_model(gpu="0"):
    global _model, _device
    use_cuda = gpu != "-1" and torch.cuda.is_available()
    _device = torch.device("cuda:0" if use_cuda else "cpu")

    if _model is not None:
        return _model, _device

    if not os.path.isfile(ADCD_CKPT):
        raise FileNotFoundError("ADCD-Net checkpoint not found: " + ADCD_CKPT)

    if not os.path.isfile(DOCRES_CKPT):
        raise FileNotFoundError("DocRes checkpoint not found: " + DOCRES_CKPT)

    import cfg
    cfg.docres_ckpt_path = DOCRES_CKPT
    cfg.val_max_size = MAX_SIZE

    from model.model import ADCDNet

    model = ADCDNet().to(_device)

    try:
        checkpoint = torch.load(ADCD_CKPT, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(ADCD_CKPT, map_location="cpu")

    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print("ADCD-Net missing keys:", len(missing))
    if unexpected:
        print("ADCD-Net unexpected keys:", len(unexpected))

    model.eval()
    _model = model
    return _model, _device


def _make_ocr_mask(image):
    import pytesseract

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    data = pytesseract.image_to_data(
        gray,
        config="--oem 1 --psm 6",
        output_type=pytesseract.Output.DICT
    )

    mask = np.zeros((h, w), dtype=np.uint8)

    for i in range(len(data["text"])):
        text = str(data["text"][i]).strip()
        if not text:
            continue

        try:
            confidence = float(data["conf"][i])
        except Exception:
            confidence = -1

        if confidence < 30:
            continue

        x = int(data["left"][i])
        y = int(data["top"][i])
        bw = int(data["width"][i])
        bh = int(data["height"][i])

        if bw > 0 and bh > 0:
            cv2.rectangle(mask, (x, y), (x + bw, y + bh), 1, -1)

    return mask


def _make_dct(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    h8 = (h // 8) * 8
    w8 = (w // 8) * 8
    gray = gray[:h8, :w8]
    dct = np.zeros((h8, w8), dtype=np.float32)

    for y in range(0, h8, 8):
        for x in range(0, w8, 8):
            block = gray[y:y + 8, x:x + 8] - 128.0
            dct[y:y + 8, x:x + 8] = cv2.dct(block)

    return np.clip(np.rint(np.abs(dct)), 0, 20).astype(np.int64)


def _prepare(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError("Could not read image: " + image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    original_h, original_w = image.shape[:2]
    original_size = (original_h, original_w)

    scale = min(MAX_SIZE / max(original_h, original_w), 1.0)

    if scale != 1.0:
        new_w = max(1, int(round(original_w * scale)))
        new_h = max(1, int(round(original_h * scale)))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    h, w = image.shape[:2]
    resized_size = (h, w)

    size = int(np.ceil(max(h, w) / 16.0) * 16)

    padded = np.zeros((size, size, 3), dtype=np.uint8)
    padded[:h, :w] = image

    ocr_mask = _make_ocr_mask(image)

    padded_ocr = np.zeros((size, size), dtype=np.uint8)
    padded_ocr[:h, :w] = ocr_mask

    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(padded, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    )

    if not ok:
        raise RuntimeError("Could not create JPEG representation for ADCD-Net.")

    jpeg_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if jpeg_bgr is None:
        raise RuntimeError("Could not decode JPEG representation for ADCD-Net.")

    jpeg_img = cv2.cvtColor(jpeg_bgr, cv2.COLOR_BGR2RGB)

    dct = _make_dct(jpeg_img)

    img_tensor = torch.from_numpy(np.asarray(jpeg_img)).permute(2, 0, 1).float() / 255.0

    mean = torch.tensor([0.485, 0.455, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    img_tensor = ((img_tensor - mean) / std).unsqueeze(0)

    dct_tensor = torch.from_numpy(dct).long().unsqueeze(0)

    qt_tensor = torch.ones((1, 8, 8), dtype=torch.long)

    ocr_tensor = torch.from_numpy(padded_ocr).long().unsqueeze(0).unsqueeze(0)

    return img_tensor, dct_tensor, qt_tensor, ocr_tensor, original_size, resized_size


def _save_outputs(image_path, probability, original_size, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    original_h, original_w = original_size
    probability = cv2.resize(probability, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

    base = os.path.splitext(os.path.basename(image_path))[0]
    heatmap_path = os.path.join(output_dir, base + "_adcd_heatmap.jpg")
    mask_path = os.path.join(output_dir, base + "_adcd_mask.png")

    heat = np.uint8(np.clip(probability, 0.0, 1.0) * 255)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    original = cv2.imread(image_path)

    if original is None:
        raise FileNotFoundError("Could not read image: " + image_path)

    overlay = cv2.addWeighted(original, 0.60, heat_color, 0.40, 0)
    cv2.imwrite(heatmap_path, overlay)

    binary = np.uint8(probability >= 0.50) * 255
    cv2.imwrite(mask_path, binary)

    return heatmap_path, mask_path, probability


def detect_tampering(image_path, output_dir, gpu="0"):
    model, device = _load_model(gpu)
    img, dct, qt, ocr_mask, original_size, resized_size = _prepare(image_path)

    img = img.to(device)
    dct = dct.to(device)
    qt = qt.to(device)
    ocr_mask = ocr_mask.to(device)

    mask = torch.zeros((1, img.shape[2], img.shape[3]), dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(img, dct, qt, mask, ocr_mask, is_train=False)

        logits = outputs[0]
        align_logits = outputs[2]

        probability = F.softmax(logits.float(), dim=1)[:, 1, :, :]
        alignment_probability = F.softmax(align_logits.float(), dim=1)[0, -1]

    probability = probability[0].detach().cpu().numpy()

    heatmap_path, mask_path, probability = _save_outputs(
        image_path, probability, original_size, output_dir
    )

    tampering_score = float(np.mean(probability))
    max_score = float(np.max(probability))
    alignment_score = float(alignment_probability.detach().cpu().item())

    result = "SUSPICIOUS" if tampering_score >= 0.50 else "LIKELY AUTHENTIC"

    return {
        "score": tampering_score,
        "max_score": max_score,
        "alignment_score": alignment_score,
        "result": result,
        "heatmap": heatmap_path,
        "mask": mask_path
    }
