#pang exlain kung bakit real or fake according sa AI
from __future__ import annotations
import os
import tempfile
import numpy as np
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt


class GradCamExplainer:
    

    CLS_LABELS = ["Fake", "Real"]
    ATTENTION_RESIZE = 14
    HEATMAP_BLUR_KERNEL = 7
    TOP_REGION_PERCENT = 0.6

    def __init__(self, model_dir: str) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_dir, output_attentions=True, ignore_mismatched_sizes=True
        ).to(self.device).eval()

    # ------------------------------------------------------------------
    def explain(self, image_path: str) -> tuple[str, str]:
        """
        Returns:  (absolute_path_to_saved_overlay, explanation_string)
        """
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            out = self.model(pixel_values, output_attentions=True, return_dict=True)
            logits = out.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            pred_idx = int(torch.argmax(probs, dim=-1).cpu().item())
            confidence = float(probs[0, pred_idx].cpu().item())
            label = self.CLS_LABELS[pred_idx]

            rollout = self._attention_rollout(out.attentions)
            heatmap = self._map_rollout_to_image(rollout, img.size[::-1])
            heatmap = self._postprocess_heatmap(heatmap)
            metrics = self._region_metrics_from_heatmap(heatmap)
            explanation = self._rule_based_explanation(label, confidence, metrics)

            overlay = self._make_overlay(img, heatmap)
            save_path = os.path.join(tempfile.gettempdir(), "gradcam_overlay.png")
            cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return os.path.abspath(save_path), explanation

    def _attention_rollout(self, attentions):
        attn_mats = [a[0].detach().cpu().numpy() for a in attentions]
        attn_heads = [np.mean(a, axis=0) for a in attn_mats]
        augmented = [a + np.eye(a.shape[0]) for a in attn_heads]
        augmented = [a / a.sum(axis=-1, keepdims=True) for a in augmented]
        rollout = augmented[0]
        for i in range(1, len(augmented)):
            rollout = augmented[i] @ rollout
        return rollout

    def _map_rollout_to_image(self, rollout, image_size):
        cls_attn = rollout[0, 1:]
        grid_size = int(np.sqrt(cls_attn.shape[0]))
        cls_map = cls_attn.reshape(grid_size, grid_size)
        cls_map = cls_map - cls_map.min()
        if cls_map.max() > 0:
            cls_map = cls_map / cls_map.max()
        h, w = image_size
        cls_map_resized = cv2.resize(cls_map, (w, h), interpolation=cv2.INTER_CUBIC)
        return cls_map_resized

    def _postprocess_heatmap(self, heatmap):
        hm = heatmap - heatmap.min()
        if hm.max() > 0:
            hm = hm / hm.max()
        k = self.HEATMAP_BLUR_KERNEL
        if k > 1:
            k = k + 1 if k % 2 == 0 else k
            hm = cv2.GaussianBlur(hm, (k, k), 0)
            hm = hm - hm.min()
            if hm.max() > 0:
                hm = hm / hm.max()
        return hm

    def _make_overlay(self, image_pil, heatmap, alpha=0.45):
        img = np.array(image_pil)
        cmap = plt.get_cmap("jet")
        colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
        overlay = cv2.addWeighted(img, 1 - alpha, colored, alpha, 0)
        return overlay

    def _region_metrics_from_heatmap(self, heatmap):
        h, w = heatmap.shape
        rows, cols = 3, 2
        rh, cw = max(1, h // rows), max(1, w // cols)
        importances = []
        for r in range(rows):
            for c in range(cols):
                patch = heatmap[r * rh:(r + 1) * rh, c * cw:(c + 1) * cw]
                importances.append(float(patch.sum()))
        total = sum(importances)
        norm = [v / total for v in importances] if total else [0.0] * len(importances)
        top_mass = sum(sorted(norm, reverse=True)[:3])
        return {"region_importances": norm, "top_region_mass": top_mass, "total_heat": total}

    def _rule_based_explanation(self, label, confidence, metrics):
        lines = []
        lines.append(f"The model predicted **{label}** with confidence {confidence * 100:.2f}%.")
        if metrics["top_region_mass"] > self.TOP_REGION_PERCENT:
            lines.append("The model's attention is strongly concentrated in a few local regions.")
        else:
            lines.append("The model's attention is distributed across larger parts of the image.")
        ri = metrics["region_importances"]
        region_names = ["top-left", "top-right", "middle-left", "middle-right", "bottom-left", "bottom-right"]
        top_idxs = sorted(range(len(ri)), key=lambda i: ri[i], reverse=True)[:2]
        top_desc = ", ".join([f"{region_names[i]} ({ri[i] * 100:.1f}%)" for i in top_idxs])
        lines.append(f"Highest attention regions: {top_desc}.")
        if label.lower() in ("fake", "ai"):
            lines.append("This often indicates local artifacts or unnatural textures in those regions.")
        else:
            lines.append("The pattern of attention supports the 'Real' classification.")
        return "\n".join(lines)