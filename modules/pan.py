import re
from modules.ocr import OCR


class PAN:
    def __init__(self):
        self.ocr = OCR()

    def extract(self, image_path):
        result = self.ocr.extract(image_path)

        if not result:
            return None

        text = self._get_text(result)
        pan_number = self._find_pan(text)

        return {
            "document_type": "PAN",
            "pan_number": pan_number,
            "valid_number": pan_number is not None,
            "raw_text": text
        }

    def _get_text(self, result):
        texts = []

        for page in result:
            data = page.json if hasattr(page, "json") else page

            if isinstance(data, dict):
                texts.extend(self._extract_strings(data))

        return " ".join(texts)

    def _extract_strings(self, data):
        values = []

        if isinstance(data, dict):
            for key, value in data.items():
                if key in ("rec_texts", "text"):
                    if isinstance(value, list):
                        values.extend(str(v) for v in value)
                    elif isinstance(value, str):
                        values.append(value)
                elif isinstance(value, (dict, list)):
                    values.extend(self._extract_strings(value))

        elif isinstance(data, list):
            for item in data:
                values.extend(self._extract_strings(item))

        return values

    def _find_pan(self, text):
        text = text.upper().replace(" ", "").replace("\n", "")
        match = re.search(r"[A-Z]{5}[0-9]{4}[A-Z]", text)
        return match.group(0) if match else None


def extract_pan(image_path):
    return PAN().extract(image_path)