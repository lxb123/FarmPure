import math
from typing import Dict, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def class_edges(one_hot: Tensor, valid: Tensor) -> Tensor:
    channels = one_hot.shape[1]
    kx = one_hot.new_tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    ky = kx.t()
    padded = F.pad(one_hot, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(padded, kx.reshape(1, 1, 3, 3).repeat(channels, 1, 1, 1), groups=channels)
    gy = F.conv2d(padded, ky.reshape(1, 1, 3, 3).repeat(channels, 1, 1, 1), groups=channels)
    neighborhood_valid = 1 - F.max_pool2d(F.pad(1 - valid, (1, 1, 1, 1), mode="replicate"), 3, 1)
    return ((gx.abs() + gy.abs()) > 0).to(one_hot.dtype) * neighborhood_valid


def soft_band(edges: Tensor, radius: int, steps: int) -> Tensor:
    radii = torch.linspace(1, radius, steps).round().to(torch.int64).unique().tolist()
    dilations = [F.max_pool2d(edges, 2 * r + 1, stride=1, padding=r) for r in radii]
    return torch.stack(dilations).mean(dim=0).amax(dim=1, keepdim=True)


class BoILLoss(nn.Module):
    def __init__(
        self,
        grid_sizes: Sequence[int] = (128,),
        band_px: int = 2,
        equivalent_thickness: Optional[float] = None,
        soft_steps: int = 3,
        max_band_weight: float = 2.0,
        warmup_epochs: int = 3,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        aux_weight: float = 0.1,
        ignore_index: int = 255,
        class_weights: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        if not grid_sizes or any(s < 1 for s in grid_sizes) or band_px < 1 or soft_steps < 1:
            raise ValueError("Grid sizes, band width and soft steps must be positive")
        if max_band_weight < 1 or warmup_epochs < 0 or min(ce_weight, dice_weight, aux_weight) < 0:
            raise ValueError("Invalid loss weights or warmup duration")
        if equivalent_thickness is not None and equivalent_thickness <= 0:
            raise ValueError("equivalent_thickness must be positive")
        if class_weights is not None and (not class_weights or min(class_weights) < 0):
            raise ValueError("class_weights must be nonempty and nonnegative")
        self.grid_sizes = tuple(grid_sizes)
        self.band_px = band_px
        self.equivalent_thickness = equivalent_thickness
        self.soft_steps = soft_steps
        self.max_band_weight = max_band_weight
        self.warmup_epochs = warmup_epochs
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index
        weights = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", weights)

    def valid_targets(self, targets: Tensor, classes: int) -> Tensor:
        if targets.ndim != 3 or targets.is_floating_point():
            raise ValueError("targets must be integer labels [B, H, W]")
        valid = targets != self.ignore_index
        values = targets[valid]
        if values.numel() and (values.min() < 0 or values.max() >= classes):
            raise ValueError("Targets contain labels outside the configured class range")
        return valid

    @torch.no_grad()
    def boundary_band(self, targets: Tensor, classes: int) -> Tensor:
        valid = self.valid_targets(targets, classes)
        height, width = targets.shape[-2:]
        bands = []
        for grid_h in self.grid_sizes:
            grid_w = max(1, round(grid_h * width / height))
            small = F.interpolate(targets[:, None].float(), (grid_h, grid_w), mode="nearest")[:, 0].long()
            small_valid = small != self.ignore_index
            safe = small.masked_fill(~small_valid, 0)
            one_hot = F.one_hot(safe, classes).permute(0, 3, 1, 2).float()
            one_hot = one_hot * small_valid[:, None]
            radius = self.band_px
            if self.equivalent_thickness is not None:
                radius = math.ceil(self.equivalent_thickness * grid_h / height)
            edges = class_edges(one_hot, small_valid[:, None].float())
            band = soft_band(edges, radius, self.soft_steps) * small_valid[:, None]
            bands.append(F.interpolate(band, (height, width), mode="bilinear", align_corners=False))
        return torch.stack(bands).amax(dim=0)[:, 0].clamp(0, 1) * valid

    def pixel_weights(self, targets: Tensor, classes: int, epoch: int) -> Tensor:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative and zero-based")
        valid = self.valid_targets(targets, classes)
        progress = 1.0 if self.warmup_epochs == 0 else min(epoch / self.warmup_epochs, 1.0)
        band = self.boundary_band(targets, classes)
        weights = (1 + (self.max_band_weight - 1) * progress * band) * valid
        mean = weights.sum() / valid.sum().clamp_min(1)
        return weights / mean.clamp_min(1e-6)

    def segmentation_loss(self, logits: Tensor, targets: Tensor, weights: Optional[Tensor] = None) -> Tensor:
        classes = max(logits.shape[1], 2)
        valid = self.valid_targets(targets, classes)
        if not valid.any():
            return logits.sum() * 0
        safe = targets.masked_fill(~valid, 0)
        if self.class_weights is not None and self.class_weights.numel() != classes:
            raise ValueError("class_weights must contain one entry per semantic class")
        if logits.shape[1] == 1:
            ce = F.binary_cross_entropy_with_logits(logits[:, 0], safe.float(), reduction="none")
            if self.class_weights is not None:
                ce = ce * self.class_weights[safe]
            probabilities = logits.sigmoid()
            one_hot = safe[:, None].float()
        else:
            ce = F.cross_entropy(logits, safe, weight=self.class_weights, reduction="none")
            probabilities = logits.softmax(dim=1)
            one_hot = F.one_hot(safe, classes).permute(0, 3, 1, 2).to(logits.dtype)
        if weights is not None:
            ce = ce * weights
        ce = ce[valid].mean()
        probabilities = probabilities * valid[:, None]
        one_hot = one_hot * valid[:, None]
        dims = (0, 2, 3)
        numerator = 2 * (probabilities * one_hot).sum(dims) + 1e-6
        denominator = probabilities.sum(dims) + one_hot.sum(dims) + 1e-6
        dice = 1 - (numerator / denominator).mean()
        return self.ce_weight * ce + self.dice_weight * dice

    def forward(self, outputs: Dict[str, Tensor], targets: Tensor, epoch: int = 0) -> Tensor:
        logits = outputs["main"]
        if logits.shape[0] != targets.shape[0] or logits.shape[-2:] != targets.shape[-2:]:
            raise ValueError("Main logits and targets must have matching batch and spatial dimensions")
        weights = self.pixel_weights(targets, max(logits.shape[1], 2), epoch)
        loss = self.segmentation_loss(logits, targets, weights)
        if "aux" in outputs and self.aux_weight > 0:
            aux = outputs["aux"]
            aux_targets = F.interpolate(targets[:, None].float(), aux.shape[-2:], mode="nearest")[:, 0].long()
            loss = loss + self.aux_weight * self.segmentation_loss(aux, aux_targets)
        return loss
