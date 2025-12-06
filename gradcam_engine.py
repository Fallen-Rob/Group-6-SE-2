# gradcam_engine.py
# ViT explainability engine using attention-rollout (works with transformers ViT)
# Produces: heatmap (numpy HxW), metrics dict, textual explanation string
#
# Requirements:
#   pip install torch torchvision transformers pillow numpy opencv-python matplotlib
#
# Usage from other code:
#   from gradcam_engine import explain_image
#   heatmap, metrics, text = explain_image(image_path="test.jpg", model_dir="./model_merged")
#
# The engine will load the PyTorch model & processor from `model_dir` (offline).
# It will use CUDA if available, otherwise CPU.

import os
import numpy as np
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# =========================
# VARIABLES YOU CAN MODIFY
# =========================
DEFAULT_MODEL_DIR = "./ai_model"    # local PyTorch model folder (must contain config + pytorch_model.bin)
CLS_LABELS = ["Fake", "Real"]           # index -> label
ATTENTION_RESIZE = 14                   # ViT patch grid size for base 224 model is usually 14 (16x16 patches)
HEATMAP_BLUR_KERNEL = 7                 # blur applied to heatmap (odd int)
TOP_REGION_PERCENT = 0.6                # used in rule-based reasoning (fraction of heat mass)
# =========================


def _load_model_and_processor(model_dir: str, device: torch.device):
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir, output_attentions=True, ignore_mismatched_sizes=True)
    model.to(device)
    model.eval()
    return model, processor


def _attention_rollout(attentions: tuple, discard_ratio=0.0):
    """
    Compute attention-rollout as in https://github.com/somepago/attn_rollout
    attentions: tuple(len_layers) of tensors shape (batch, n_heads, seq_len, seq_len)
    returns: attention map shape (seq_len, seq_len)
    """
    # Convert to numpy tensors (single batch)
    # attentions is tuple of tensors: use first element in batch if batch>1
    attn_mats = [a[0].detach().cpu().numpy() for a in attentions]  # list of (n_heads, seq, seq)
    n_layers = len(attn_mats)
    # average heads
    attn_heads = [np.mean(a, axis=0) for a in attn_mats]  # list of (seq, seq)
    # add identity and normalize
    augmented = [a + np.eye(a.shape[0]) for a in attn_heads]
    augmented = [a / a.sum(axis=-1, keepdims=True) for a in augmented]
    # rollout: matrix multiply
    rollout = augmented[0]
    for i in range(1, len(augmented)):
        rollout = augmented[i].dot(rollout)
    return rollout  # shape (seq, seq)


def _map_rollout_to_image(rollout, image_size: tuple, patch_size: int = 16):
    """
    rollout: (seq, seq) where seq = num_patches + 1 (CLS)
    We compute the attention from CLS token to patches: rollout[0, 1:]
    Then reshape to patch grid and upscale to image size.
    """
    seq_len = rollout.shape[0]
    # exclude CLS token index 0
    cls_attn = rollout[0, 1:]  # shape (num_patches,)
    num_patches = cls_attn.shape[0]
    grid_size = int(np.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        # fallback: try ATTENTION_RESIZE
        grid_size = ATTENTION_RESIZE
        if grid_size * grid_size != num_patches:
            # as last resort, compute grid size via ceil
            grid_size = int(np.sqrt(num_patches))
    cls_map = cls_attn.reshape(grid_size, grid_size)
    # normalize
    cls_map = cls_map - cls_map.min()
    if cls_map.max() > 0:
        cls_map = cls_map / cls_map.max()
    # upscale to image size using cv2
    H, W = image_size
    cls_map_resized = cv2.resize(cls_map, (W, H), interpolation=cv2.INTER_CUBIC)
    return cls_map_resized


def _postprocess_heatmap(heatmap, blur_kernel=HEATMAP_BLUR_KERNEL):
    # normalize 0..1 and blur
    hm = heatmap - heatmap.min()
    if hm.max() > 0:
        hm = hm / hm.max()
    if blur_kernel > 1:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        hm = cv2.GaussianBlur(hm, (blur_kernel, blur_kernel), 0)
        # renormalize
        hm = hm - hm.min()
        if hm.max() > 0:
            hm = hm / hm.max()
    return hm


def _make_overlay(image_pil: Image.Image, heatmap: np.ndarray, alpha=0.5):
    img = np.array(image_pil.convert("RGB"))
    cmap = plt.get_cmap("jet")
    colored = cmap(heatmap)[:, :, :3]  # HxWx3
    colored = (colored * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img, 1 - alpha, colored, alpha, 0)
    return overlay


def _region_metrics_from_heatmap(heatmap: np.ndarray, n_regions=6):
    """
    Divide image into simple regions (grid) and compute % heat mass per region
    returns dict with total_heat, top_region_fraction, region_importances (list)
    """
    H, W = heatmap.shape
    # define 3x2 grid regions (you can change)
    rows = 3
    cols = 2
    rh = H // rows
    cw = W // cols
    importances = []
    for r in range(rows):
        for c in range(cols):
            y0 = r * rh
            x0 = c * cw
            patch = heatmap[y0:y0 + rh, x0:x0 + cw]
            importances.append(float(patch.sum()))
    total = float(np.sum(importances))
    if total > 0:
        norm = [v / total for v in importances]
    else:
        norm = [0.0 for _ in importances]
    # sort and top region mass
    sorted_norm = sorted(norm, reverse=True)
    top_mass = sum(sorted_norm[:3])  # mass of top 3 regions
    return {"region_importances": norm, "top_region_mass": top_mass, "total_heat": total}


def _rule_based_explanation(pred_label: str, confidence: float, metrics: dict):
    """
    Simple rule-based template to generate a human-readable explanation.
    """
    lines = []
    lines.append(f"The model predicted **{pred_label}** with confidence {confidence*100:.2f}%.")
    top_mass = metrics.get("top_region_mass", 0.0)
    if top_mass > TOP_REGION_PERCENT:
        lines.append("The model's attention is strongly concentrated in a few local regions.")
    else:
        lines.append("The model's attention is distributed across larger parts of the image.")
    # region importance: convert to friendly text
    ri = metrics.get("region_importances", [])
    if ri:
        # map region indices to human terms (approx)
        region_names = [
            "top-left", "top-right",
            "middle-left", "middle-right",
            "bottom-left", "bottom-right"
        ]
        # find top 2 regions
        top_idxs = sorted(range(len(ri)), key=lambda i: ri[i], reverse=True)[:2]
        top_desc = ", ".join([f"{region_names[i]} ({ri[i]*100:.1f}%)" for i in top_idxs])
        lines.append(f"Highest attention regions: {top_desc}.")
    # closing hypothesis
    if pred_label.lower() in ("fake", "ai"):
        lines.append("This often indicates local artifacts or unnatural textures in those regions.")
    else:
        lines.append("The pattern of attention supports the 'Real' classification.")
    return "\n".join(lines)


def explain_image(image_path: str,
                  model_dir: str = DEFAULT_MODEL_DIR,
                  device: str = None,  # type: ignore
                  save_overlay: str = None):  # type: ignore
    """
    Main entry point.
    Returns: (heatmap HxW np array, metrics dict, explanation string, overlay image array if saved)
    """
    # device handling
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)  # type: ignore

    # load
    model, processor = _load_model_and_processor(model_dir, device)  # type: ignore

    # load image
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size[0], img.size[1]

    # preprocess via processor - we will use processor to get pixel_values tensor for model
    inputs = processor(img, return_tensors="pt")  # returns numpy or torch depending, we want torch
    # ensure torch tensors and move to device
    if isinstance(inputs["pixel_values"], torch.Tensor):
        pixel_values = inputs["pixel_values"].to(device)
    else:
        pixel_values = torch.tensor(inputs["pixel_values"]).to(device)

    # forward pass, ask for attentions
    with torch.no_grad():
        outputs = model(pixel_values, output_attentions=True, return_dict=True)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred_idx = int(torch.argmax(probs, dim=-1).cpu().item())
        pred_label = CLS_LABELS[pred_idx] if pred_idx < len(CLS_LABELS) else str(pred_idx)
        confidence = float(probs[0, pred_idx].cpu().item())

        # attentions: tuple of (n_layers, batch, n_heads, seq_len, seq_len)
        if outputs.attentions is None:
            raise RuntimeError("Model did not return attentions. Make sure model supports output_attentions=True.")
        rollout = _attention_rollout(outputs.attentions)  # shape (seq, seq)
        heatmap = _map_rollout_to_image(rollout, image_size=(orig_h, orig_w))
        heatmap = _postprocess_heatmap(heatmap)
        metrics = _region_metrics_from_heatmap(heatmap)
        explanation = _rule_based_explanation(pred_label, confidence, metrics)

    # create overlay image (RGB numpy)
    overlay = _make_overlay(img, heatmap, alpha=0.45)
    if save_overlay:
        # save overlay (PNG)
        cv2.imwrite(save_overlay, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return heatmap, metrics, explanation, overlay