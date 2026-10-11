import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from farmpure import FarmPure, FrozenDINOv3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/reference.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    encoder = FrozenDINOv3.from_pretrained(config["encoder_path"], config["layers"])
    model = FarmPure(encoder, **config["model"]).to(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = model.load_state_dict(state, strict=False)
    if any(not key.startswith("encoder.") for key in state.missing_keys) or state.unexpected_keys:
        raise ValueError(f"Checkpoint does not match this implementation: {state}")
    model.eval()
    with Image.open(args.image) as source:
        original_size = source.size
        height, width = config["image_size"]
        image = source.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        tensor = torch.tensor(bytearray(image.tobytes()), dtype=torch.uint8)
        tensor = tensor.reshape(height, width, 3).permute(2, 0, 1).float() / 255
    with torch.no_grad():
        logits = model(tensor[None].to(args.device))["main"]
        if logits.shape[1] == 1:
            mask = (logits[0, 0].sigmoid() >= 0.5).to(torch.uint8) * 255
        else:
            if logits.shape[1] > 255:
                raise ValueError("This PNG example supports at most 255 classes")
            mask = logits[0].argmax(dim=0).to(torch.uint8)
    output = Image.frombytes("L", (width, height), bytes(mask.cpu().flatten().tolist()))
    output = output.resize(original_size, Image.Resampling.NEAREST)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)
    print(f"Saved mask: {args.output}")


if __name__ == "__main__":
    main()
