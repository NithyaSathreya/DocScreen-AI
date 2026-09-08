import argparse
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Visualize TruFor tampering map.")
    parser.add_argument("--input", required=True, help="Original image.")
    parser.add_argument("--result", required=True, help="TruFor .npz result.")
    parser.add_argument("--output", required=True, help="Output heatmap image.")
    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")

    data = np.load(args.result)
    score = float(data["score"])
    pred = cv2.resize(data["map"], (image.shape[1], image.shape[0]))

    heatmap = np.uint8(np.clip(pred, 0, 1) * 255)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.60, heatmap, 0.40, 0)

    cv2.imwrite(args.output, overlay)

    print(f"TruFor score: {score:.4f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()