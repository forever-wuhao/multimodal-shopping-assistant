"""Frozen OpenCLIP backbone plus an optional locally trained residual adapter."""
import hashlib
import json
from pathlib import Path

import numpy as np


def normalize(values):
    values = np.asarray(values, dtype="float32")
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite 2D embeddings")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms < 1e-8).any():
        raise ValueError("Zero-length embedding")
    return np.ascontiguousarray(values / norms)


def load_adapter(path):
    with np.load(path, allow_pickle=False) as file:
        metadata = json.loads(file["metadata"].tobytes())
        down, up = file["down"].copy(), file["up"].copy()
    if metadata.get("schema") != 1 or down.ndim != 2 or up.shape != down.T.shape:
        raise ValueError("Invalid adapter shape/schema")
    if not np.isfinite(down).all() or not np.isfinite(up).all():
        raise ValueError("Non-finite adapter weights")
    return metadata, down, up


class ClipEncoder:
    def __init__(self, model="ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu", adapter=None):
        import open_clip
        import torch
        self.torch = torch
        self.device = device
        self.config = {"model": model, "pretrained": pretrained}
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained, device=device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model)
        self.adapter = None
        self.adapter_metadata = None
        self.adapter_sha = None
        if adapter:
            metadata, down, up = load_adapter(adapter)
            if metadata["encoder"] != self.config or down.shape[1] != self.model.visual.output_dim:
                raise ValueError("Adapter and OpenCLIP backbone do not match")
            self.adapter = down, up
            self.adapter_metadata = metadata
            self.adapter_sha = hashlib.sha256(Path(adapter).read_bytes()).hexdigest()

    def project(self, features):
        features = normalize(features)
        if self.adapter:
            down, up = self.adapter
            features = features + np.maximum(features @ down.T, 0) @ up.T
        return normalize(features)

    def images(self, images):
        batch = self.torch.stack([self.preprocess(image) for image in images]).to(self.device)
        with self.torch.inference_mode():
            features = self.model.encode_image(batch).float().cpu().numpy()
        return self.project(features)

    def texts(self, texts):
        with self.torch.inference_mode():
            features = self.model.encode_text(self.tokenizer(texts).to(self.device)).float().cpu().numpy()
        return self.project(features)

    def identity(self):
        return {**self.config, "adapter_sha256": self.adapter_sha}
