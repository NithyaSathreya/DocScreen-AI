import re
from modules.ocr import OCR


class Aadhaar:
    def __init__(self):
        self.ocr = OCR()

    def extract(self, image_path):
        result = self.ocr.extract(image_path)

        if not result:
            return None

        text = self._get_text(result)
        aadhaar_number = self._find_aadhaar(text)

        return {
            "document_type": "AADHAAR",
            "aadhaar_number": aadhaar_number,
            "valid_number": aadhaar_number is not None,
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

    def _find_aadhaar(self, text):
        text = text.replace("-", " ").replace("\n", " ")
        matches = re.findall(r"\b\d{4}\s*\d{4}\s*\d{4}\b", text)

        if not matches:
            return None

        return re.sub(r"\D", "", matches[0])


def extract_aadhaar(image_path):
    return Aadhaar().extract(image_path)