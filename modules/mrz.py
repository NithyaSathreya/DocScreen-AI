from passporteye import read_mrz


class MRZ:
    def extract(self, image_path):
        result = read_mrz(image_path)
        if result is None:
            return None

        data = result.to_dict()

        return {
            "document_type": data.get("type"),
            "country": data.get("country"),
            "passport_number": data.get("number"),
            "nationality": data.get("nationality"),
            "date_of_birth": data.get("date_of_birth"),
            "expiration_date": data.get("expiration_date"),
            "sex": data.get("sex"),
            "surname": data.get("surname"),
            "given_names": data.get("names"),
            "personal_number": data.get("personal_number"),
            "valid_number": data.get("valid_number", False),
            "valid_date_of_birth": data.get("valid_date_of_birth", False),
            "valid_expiration_date": data.get("valid_expiration_date", False),
            "valid_personal_number": data.get("valid_personal_number", False),
            "valid_composite": data.get("valid_composite", False),
            "valid_score": data.get("valid_score", 0),
            "raw_text": data.get("raw_text")
        }


def extract_mrz(image_path):
    return MRZ().extract(image_path)