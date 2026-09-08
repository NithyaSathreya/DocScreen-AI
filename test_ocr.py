import argparse
from modules.ocr import extract_text


def main():
    parser = argparse.ArgumentParser(description="OCR document text extraction.")
    parser.add_argument("--input", required=True, help="Path to document image.")
    parser.add_argument("--lang", default="en", help="OCR language.")
    args = parser.parse_args()

    result = extract_text(args.input, args.lang)
    print(result)


if __name__ == "__main__":
    main()