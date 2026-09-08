import argparse
from modules.mrz import extract_mrz


def main():
    parser = argparse.ArgumentParser(description="Passport MRZ extraction test.")
    parser.add_argument("--input", required=True, help="Path to passport image.")
    args = parser.parse_args()

    result = extract_mrz(args.input)

    if result is None:
        print("No MRZ detected.")
        return

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()