# gradcam_preview.py
# CLI tool to preview attention heatmap and textual explanation for a single image
#
# Usage:
#   python gradcam_preview.py --image test.jpg --model_dir ./model_merged --out overlay.png

import argparse
import os
from gradcam_engine import explain_image
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model_dir", default="./model_merged", help="Local PyTorch model folder")
    parser.add_argument("--out", default="overlay_out.png", help="Path to save overlay image")
    parser.add_argument("--show", action="store_true", help="Display overlay in a window")
    args = parser.parse_args()

    hm, metrics, explanation, overlay = explain_image(args.image, model_dir=args.model_dir, save_overlay=args.out)
    print("=== Explanation ===")
    print(explanation)
    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"\nOverlay saved to: {args.out}")

    if args.show:
        # show via matplotlib
        plt.figure(figsize=(8, 8))
        plt.imshow(overlay)
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    main()