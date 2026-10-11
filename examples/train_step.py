import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from farmpure import BoILLoss, FarmPure, FrozenDINOv3


class ToyBackbone(nn.Module):
    def __init__(self, depth: int = 2, channels: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=channels, patch_size=16, num_register_tokens=0, num_hidden_layers=depth
        )
        self.patch_embed = nn.Conv2d(3, channels, 16, stride=16)
        self.blocks = nn.ModuleList([nn.Linear(channels, channels) for _ in range(depth)])

    def forward(self, pixel_values: torch.Tensor, output_hidden_states: bool = True) -> SimpleNamespace:
        patches = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        tokens = torch.cat([patches.new_zeros(patches.shape[0], 1, patches.shape[2]), patches], dim=1)
        states = [tokens]
        for block in self.blocks:
            tokens = tokens + block(tokens).tanh()
            states.append(tokens)
        return SimpleNamespace(hidden_states=tuple(states))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.json"))
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epoch", type=int, default=3)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    torch.manual_seed(42)
    if args.mock:
        encoder = FrozenDINOv3(ToyBackbone(depth=max(config["layers"])), config["layers"])
    else:
        encoder = FrozenDINOv3.from_pretrained(config["encoder_path"], config["layers"])
    model = FarmPure(encoder, **config["model"]).to(args.device).train()
    criterion = BoILLoss(**config["loss"]).to(args.device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    height, width = config["image_size"]
    images = torch.rand(2, 3, height, width, device=args.device)
    targets = torch.zeros(2, height, width, dtype=torch.long, device=args.device)
    targets[:, height // 4:3 * height // 4, width // 4:3 * width // 4] = 1
    optimizer.zero_grad(set_to_none=True)
    outputs = model(images)
    loss = criterion(outputs, targets, epoch=args.epoch)
    loss.backward()
    assert all(p.grad is None for p in encoder.parameters())
    assert any(p.grad is not None for p in parameters)
    optimizer.step()
    print(f"Synthetic training step: loss={loss.item():.6f}, main_shape={tuple(outputs['main'].shape)}")


if __name__ == "__main__":
    main()
