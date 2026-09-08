import argparse
from modules.face import detect_faces


def main():
    parser = argparse.ArgumentParser(description="Face detection test.")
    parser.add_argument("--input", required=True, help="Path to image.")
    args = parser.parse_args()

    faces = detect_faces(args.input)
    print(f"Faces detected: {len(faces)}")

    for i, face in enumerate(faces, 1):
        print(f"Face {i}: bbox={face.bbox.astype(int).tolist()}, score={face.det_score:.4f}")


if __name__ == "__main__":
    main()