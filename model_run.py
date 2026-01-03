#basically utak nung ai 
import os
import sys
import numpy as np
import torch
import onnxruntime as ort # type: ignore
from transformers import AutoImageProcessor 
from PIL import Image
from gradcam_engine import GradCamExplainer 


class OnnxInference:
  #load nung onnx
    def __init__(self, model_dir: str = "ai_model") -> None:
        self.model_dir = self._resource_path(model_dir)
        onnx_path = os.path.join(self.model_dir, "model_ai-generated_opt.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(onnx_path)

        # ONNX session
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, so, providers=providers)

        # Processor
        self.processor = AutoImageProcessor.from_pretrained(self.model_dir)

        # Labels
        self.labels = ["fake", "real"]

        # Grad-CAM engine (lazy-loaded on first use)
        self._gradcam: GradCamExplainer | None = None

    # ------------------------------------------------------------------
    def predict(self, image_path: str) -> tuple[str, float]:
        """
        Returns:  (label: str, confidence: float 0-1)
        """
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(img, return_tensors="np")
        pixel_values = inputs["pixel_values"].astype(np.float32)

        outputs = self.session.run(None, {self.session.get_inputs()[0].name: pixel_values})
        logits = torch.tensor(outputs[0])
        probs = torch.nn.functional.softmax(logits, dim=-1)
        label_id = int(torch.argmax(probs).item())
        confidence = float(probs[0, label_id])
        return self.labels[label_id], confidence

    def predict_with_gradcam(self, image_path: str) -> tuple[str, float, str, str]:
        label, conf = self.predict(image_path)  # reuse preprocessing
        if self._gradcam is None:
            self._gradcam = GradCamExplainer(self.model_dir)
        heatmap_path, explanation = self._gradcam.explain(image_path)
        return label, conf, heatmap_path, explanation

    @staticmethod
    def _resource_path(rel: str) -> str:
        """PyInstaller one-file bundle helper."""
        if hasattr(sys, "_MEIPASS"):
            return os.path.join(sys._MEIPASS, rel)
        return os.path.join(os.path.abspath("."), rel)