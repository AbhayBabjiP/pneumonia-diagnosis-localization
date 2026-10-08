"""DenseNet121 classifier and Grad-CAM utilities."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121


def build_model(pretrained: bool = False) -> nn.Module:
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model


class GradCAM:
    def __init__(self, model: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer = model.features.norm5
        self.handle = target_layer.register_forward_hook(self._forward_hook)
        self.grad_handle = None

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient):
        self.gradients = gradient

    def __call__(self, image: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        score = self.model(image)
        if target_class is None:
            target_class = int((score.detach().sigmoid() >= 0.5).item())
        scalar = score.sum() if target_class == 1 else -score.sum()
        scalar.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    def close(self):
        self.handle.remove()
