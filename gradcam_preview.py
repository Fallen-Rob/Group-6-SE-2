
import argparse
import os
import sys
from gradcam_engine import GradCamExplainer
import matplotlib.pyplot as plt


class GradCamPreviewCLI:

    @staticmethod
    def run(image: str, model_dir: str, out: str, show: bool = False) -> None:
        explainer = GradCamExplainer(model_dir)
        save_path, explanation = explainer.explain(image)
        if out != os.path.basename(save_path):
            import shutil
            shutil.move(save_path, out)
            save_path = os.path.abspath(out)
        print("=== Explanation ===")
        print(explanation)
        print("\nOverlay saved to:", save_path)
        if show:
            img = plt.imread(save_path)
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.axis("off")
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model-dir", default="ai_model")
    parser.add_argument("--out", default="overlay_out.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    GradCamPreviewCLI.run(args.image, args.model_dir, args.out, args.show)