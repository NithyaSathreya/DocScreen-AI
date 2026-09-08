from pathlib import Path
from paddleocr import PaddleOCR


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "networks" / "paddleocr"


class OCR:
    def __init__(self, lang="en"):
        self.engine = PaddleOCR(
            text_detection_model_dir=str(MODEL_DIR / "PP-OCRv6_medium_det"),
            text_recognition_model_dir=str(MODEL_DIR / "PP-OCRv6_medium_rec"),
            device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
            text_det_limit_side_len=960
        )

    def extract(self, image_path):
        return list(self.engine.predict(image_path))


def extract_text(image_path, lang="en"):
    return OCR(lang).extract(image_path)