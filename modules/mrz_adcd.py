import cv2
import numpy as np
import pytesseract


def _find_mrz_region(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Passport MRZ is normally located in the lower part of the document.
    start = int(h * 0.55)
    roi = gray[start:h, :]

    # Improve contrast and make characters easier for OCR.
    scale = 2
    roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return roi, start, scale


def _clean_mrz_text(text):
    lines = []

    for line in text.splitlines():
        line = line.strip().upper()
        line = "".join(c for c in line if c.isalnum() or c == "<")

        if len(line) >= 20:
            lines.append(line)

    return lines


def extract_mrz_adcd(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError("Could not read image: " + image_path)

    roi, start, scale = _find_mrz_region(image)

    config = "--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"

    text = pytesseract.image_to_string(roi, config=config)
    lines = _clean_mrz_text(text)

    # Prefer the two longest MRZ-like lines.
    candidates = sorted(lines, key=len, reverse=True)

    if len(candidates) < 2:
        return {
            "status": "not_detected",
            "raw_text": text,
            "lines": lines
        }

    mrz_lines = candidates[:2]

    # Keep the original order from the OCR output.
    selected = [line for line in lines if line in mrz_lines][:2]

    return {
        "status": "success",
        "lines": selected,
        "raw_text": text,
        "region": {
            "y_start": start,
            "y_end": image.shape[0],
            "scale": scale
        }
    }
