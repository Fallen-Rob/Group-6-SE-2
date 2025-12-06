# updated_run_onnx_with_gradcam.py
# ONNX inference + automatic Grad-CAM explanation (attention-rollout)
# Uses GPU if available for ONNX and PyTorch. Fully offline.

import onnxruntime as ort # type: ignore
from transformers import AutoImageProcessor
from PIL import Image
import numpy as np
import torch
import sys
from gradcam_engine import explain_image

# =====================================================
# VARIABLES YOU CAN MODIFY
# =====================================================
ONNX_MODEL_PATH = "./ai_model/model_ai-generated.onnx"    # optimized ONNX model
MODEL_DIR ="./ai_model"                              # PyTorch model + processor folder                              # default image
HEATMAP_OUTPUT = "gradcam_overlay.png"                     # saved overlay from Grad-CAM
LABELS = ["Fake", "Real"]                                 # index → class
# =====================================================


def choose_providers():
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_session(path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = choose_providers()
    session = ort.InferenceSession(path, so, providers=providers)
    print("\nONNX providers used:", session.get_providers())
    return session


def run_onnx(image_path):
    processor = AutoImageProcessor.from_pretrained(MODEL_DIR)

    img = Image.open(image_path).convert("RGB")
    inputs = processor(img, return_tensors="np")

    session = load_session(ONNX_MODEL_PATH)
    inp_name = session.get_inputs()[0].name

    pixel_values = inputs["pixel_values"].astype(np.float32)

    outputs = session.run(None, {inp_name: pixel_values})
    logits = torch.tensor(outputs[0])
    probs = torch.nn.functional.softmax(logits, dim=-1)

    label_id = int(torch.argmax(probs, dim=-1).cpu().item())
    confidence = float(probs[0, label_id])

    return LABELS[label_id], confidence


def run_gradcam(image_path):
    print("\nRunning Grad-CAM explanation (PyTorch)... This may take a few seconds...\n")

    heatmap, metrics, explanation, overlay = explain_image(
        image_path=image_path,
        model_dir=MODEL_DIR,
        save_overlay=HEATMAP_OUTPUT
    )

    print("=== Grad-CAM Explanation ===")
    print(explanation)

    print(f"\nHeatmap saved to: {HEATMAP_OUTPUT}")

    return HEATMAP_OUTPUT, explanation
