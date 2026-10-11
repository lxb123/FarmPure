# FarmPure

FarmPure is an image segmentation model built with a frozen DINOv3 ViT encoder, Calibrated Multi-Scale Fusion (CMSF), and Boundary-Oriented Invariance Learning (BoIL).

## Repository Layout

```text
configs/reference.json    # Standard architecture settings
configs/smoke.json        # Settings for small synthetic tests
farmpure/encoder.py       # DINOv3 encoder adapter
farmpure/model.py         # CMSF and segmentation decoder
farmpure/losses.py        # BoIL and segmentation losses
examples/train_step.py    # Training step example
examples/predict.py       # Single-image prediction script
tests/test_core.py        # Core behavior tests
```

## Installation

Requires **Python 3.10+**. Install your environment-specific PyTorch build, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

*(Note: Native DINOv3 support requires `transformers >= 4.56`)*

## Usage

### 1. Quick Test (Mock Encoder)

Verify your environment by running the unit tests and a synthetic training step without needing real weights:

```bash
python -m unittest discover -s tests -v
python -m examples.train_step --config configs/smoke.json --mock
```

### 2. Training

Download the **DINOv3 ViT-L/16 SAT-493M** checkpoint from the [official Meta Hugging Face model page](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m). Request access on the model page before downloading. Update your configuration file (`encoder_path`) to point to the local Hugging Face Transformers directory containing the configuration and weights.

Official resources: [DINOv3 repository](https://github.com/facebookresearch/dinov3) | [Meta weight download portal](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) | [ViT-L/16 LVD-1689M (web-pretrained)](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m).

Run a training step:

```bash
python -m examples.train_step --config configs/reference.json --device cuda
```

*Input tensors must be float values in `[0, 1]`. Image dimensions must be multiples of 16.*

### 3. Inference

To run predictions, you should use a saved decoder checkpoint. When saving weights during your own training loop, exclude the frozen encoder:

```python
torch.save(
    {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("encoder.")},
    "decoder.pt",
)
```

Run inference on a single image:

```bash
python -m examples.predict \
  --config configs/reference.json \
  --checkpoint decoder.pt \
  --image sample.png \
  --output outputs/mask.png \
  --device cuda
```

## Configuration

Settings are managed via JSON files (e.g., `configs/reference.json`).

**Key Parameters:**

*   `encoder_path`: Local path to the DINOv3 ViT checkpoint.
*   `model.num_classes`: 1 for binary classification, or `C` for multiclass.
*   `model.use_cmsf`: Boolean to enable/disable CMSF.
*   `loss.ce_weight` / `loss.dice_weight` / `loss.aux_weight`: Loss coefficients.
*   `loss.ignore_index`: Label index to exclude from loss computations.
*   `loss.max_band_weight`: Maximum boundary weight for BoIL (1 disables emphasis).
