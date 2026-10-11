from pathlib import Path
from typing import List, Sequence, Union

import torch
from torch import Tensor, nn


class FrozenDINOv3(nn.Module):
    def __init__(self, backbone: nn.Module, layers: Sequence[int]) -> None:
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.layers = tuple(layers)
        config = backbone.config
        self.embed_dim = int(config.hidden_size)
        self.patch_size = int(config.patch_size)
        self.special_tokens = 1 + int(config.num_register_tokens)
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("layers must be nonempty and unique")
        if any(layer < 1 or layer > config.num_hidden_layers for layer in self.layers):
            raise ValueError("layers must be one-based transformer block indices")
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @classmethod
    def from_pretrained(cls, path: Union[str, Path], layers: Sequence[int]) -> "FrozenDINOv3":
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained(str(path), local_files_only=True)
        if backbone.config.model_type != "dinov3_vit":
            raise ValueError("Expected a Hugging Face DINOv3 ViT checkpoint")
        return cls(backbone, layers)

    def train(self, mode: bool = True) -> "FrozenDINOv3":
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, images: Tensor) -> List[Tensor]:
        if images.ndim != 4 or images.shape[1] != 3 or not images.is_floating_point():
            raise ValueError("images must be float RGB tensors with shape [B, 3, H, W]")
        height, width = images.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("Image dimensions must be multiples of the encoder patch size")
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("images must contain finite values in [0, 1]")
        outputs = self.backbone(pixel_values=(images - self.mean) / self.std, output_hidden_states=True)
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        features = []
        for layer in self.layers:
            tokens = outputs.hidden_states[layer]
            expected = self.special_tokens + grid_h * grid_w
            if tokens.shape[1:] != (expected, self.embed_dim):
                raise ValueError("Unexpected token layout; no token padding or truncation is performed")
            patches = tokens[:, self.special_tokens:]
            features.append(patches.transpose(1, 2).reshape(images.shape[0], self.embed_dim, grid_h, grid_w))
        return features
