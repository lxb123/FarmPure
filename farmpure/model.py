from typing import Dict, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .encoder import FrozenDINOv3


def conv_block(in_channels: int, out_channels: int, kernel_size: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ChannelCalibration(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 8, 16)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: Tensor) -> Tensor:
        return features * self.gate(features)


class CMSF(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                ChannelCalibration(channels),
            ) for d in (1, 3, 5)
        ])
        self.selection = nn.Conv2d(3 * channels, 3, 1)
        self.projection = nn.Conv2d(channels, channels, 1)

    def scale_weights(self, branches: Sequence[Tensor]) -> Tensor:
        return self.selection(torch.cat(branches, dim=1)).softmax(dim=1)

    def forward(self, features: Tensor) -> Tensor:
        branches = [branch(features) for branch in self.branches]
        weights = self.scale_weights(branches)
        fused = sum(weights[:, i:i + 1] * branch for i, branch in enumerate(branches))
        return self.projection(fused)


def upsample_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class FarmPure(nn.Module):
    def __init__(
        self,
        encoder: FrozenDINOv3,
        num_classes: int = 1,
        fusion_dim: int = 512,
        decoder_channels: Sequence[int] = (512, 256, 128),
        use_cmsf: bool = True,
    ) -> None:
        super().__init__()
        if encoder.patch_size != 16:
            raise ValueError("This decoder expects a DINOv3 ViT with patch size 16")
        if len(decoder_channels) != 3 or min(decoder_channels) < 1 or min(num_classes, fusion_dim) < 1:
            raise ValueError("Expected three positive decoder widths and positive model dimensions")
        self.encoder = encoder
        self.fusion = nn.Sequential(
            conv_block(encoder.embed_dim * len(encoder.layers), encoder.embed_dim),
            conv_block(encoder.embed_dim, fusion_dim, 3),
        )
        self.cmsf = CMSF(fusion_dim) if use_cmsf else nn.Identity()
        c1, c2, c3 = decoder_channels
        self.stage1 = nn.Sequential(conv_block(fusion_dim, c1), upsample_block(c1, c2))
        self.stage2 = upsample_block(c2, c3)
        self.stage3 = nn.Sequential(upsample_block(c3, c3), upsample_block(c3, c3))
        self.main_head = nn.Conv2d(c3, num_classes, 1)
        self.aux_head = nn.Conv2d(c3, num_classes, 1)

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        features = self.encoder(images)
        size = features[-1].shape[-2:]
        aligned = [F.interpolate(x, size=size, mode="bilinear", align_corners=False) for x in features]
        fused = self.cmsf(self.fusion(torch.cat(aligned, dim=1)))
        intermediate = self.stage2(self.stage1(fused))
        result = {"main": self.main_head(self.stage3(intermediate))}
        if self.training:
            result["aux"] = self.aux_head(intermediate)
        return result
