import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from examples.predict import main as predict
from examples.train_step import ToyBackbone
from farmpure import BoILLoss, CMSF, FarmPure, FrozenDINOv3
from farmpure.losses import soft_band


class CoreTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def make_model(self, classes: int = 1, use_cmsf: bool = True) -> FarmPure:
        encoder = FrozenDINOv3(ToyBackbone(), (1, 2))
        return FarmPure(encoder, classes, 16, (16, 8, 4), use_cmsf)

    def test_training_and_inference(self) -> None:
        model = self.make_model().train()
        images = torch.rand(2, 3, 32, 48)
        outputs = model(images)
        self.assertEqual(outputs["main"].shape, (2, 1, 32, 48))
        self.assertEqual(outputs["aux"].shape, (2, 1, 8, 12))
        loss = BoILLoss(grid_sizes=(8, 16))(outputs, torch.randint(0, 2, (2, 32, 48)), epoch=3)
        loss.backward()
        self.assertFalse(model.encoder.backbone.training)
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        self.assertIsNotNone(model.cmsf.selection.weight.grad)
        self.assertIsNotNone(model.aux_head.weight.grad)
        optimizer_ids = {id(p) for p in model.parameters() if p.requires_grad}
        self.assertTrue(all(id(p) not in optimizer_ids for p in model.encoder.parameters()))
        model.eval()
        with torch.no_grad():
            self.assertEqual(set(model(images)), {"main"})

    def test_multiclass_and_plain(self) -> None:
        model = self.make_model(3, use_cmsf=False)
        targets = torch.randint(0, 3, (2, 32, 32))
        targets[:, :4] = 255
        criterion = BoILLoss(grid_sizes=(8, 16), class_weights=(1.0, 2.0, 1.0))
        loss = criterion(model(torch.rand(2, 3, 32, 32)), targets, epoch=2)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_selection_is_convex(self) -> None:
        cmsf = CMSF(8)
        x = torch.randn(2, 8, 4, 6)
        weights = cmsf.scale_weights([b(x) for b in cmsf.branches])
        torch.testing.assert_close(weights.sum(1), torch.ones(2, 4, 6))
        self.assertTrue((weights >= 0).all())

    def test_constant_labels_have_no_edges(self) -> None:
        criterion = BoILLoss(grid_sizes=(8, 16))
        for value in (0, 1, 255):
            band = criterion.boundary_band(torch.full((2, 16, 16), value), 2)
            self.assertEqual(band.count_nonzero().item(), 0)

    def test_soft_band_averages_unique_radii(self) -> None:
        edges = torch.zeros(1, 1, 9, 9)
        edges[:, :, 4, 4] = 1
        band = soft_band(edges, 2, 3)
        self.assertEqual(band[0, 0, 4, 4].item(), 1)
        self.assertEqual(band[0, 0, 4, 6].item(), 0.5)
        self.assertEqual(band[0, 0, 4, 7].item(), 0)

    def test_weights_and_warmup(self) -> None:
        criterion = BoILLoss(grid_sizes=(8, 16), equivalent_thickness=4)
        targets = torch.zeros(2, 16, 24, dtype=torch.long)
        targets[:, 4:12, 8:16] = 1
        targets[:, :2] = 255
        valid = targets != 255
        initial = criterion.pixel_weights(targets, 2, 0)
        torch.testing.assert_close(initial[valid], torch.ones_like(initial[valid]))
        band = criterion.boundary_band(targets, 2)
        self.assertTrue(((band >= 0) & (band <= 1)).all())
        self.assertGreater(band.sum().item(), 0)
        final = criterion.pixel_weights(targets, 2, 3)
        self.assertEqual(final[~valid].count_nonzero().item(), 0)
        torch.testing.assert_close(final[valid].mean(), torch.tensor(1.0))
        self.assertGreater(final[valid].max().item(), final[valid].min().item())
        torch.testing.assert_close(final, criterion.pixel_weights(targets, 2, 10))

    def test_ignore_does_not_change_loss(self) -> None:
        for channels in (1, 3):
            criterion = BoILLoss(grid_sizes=(8,))
            targets = torch.zeros(2, 16, 16, dtype=torch.long)
            targets[:, 4:12, 4:12] = 1
            targets[:, :2] = 255
            logits = torch.randn(2, channels, 16, 16, requires_grad=True)
            changed = logits.detach().clone()
            changed[:, :, :2] = 100
            original = criterion({"main": logits}, targets, 3)
            torch.testing.assert_close(original, criterion({"main": changed}, targets, 3))
            original.backward()
            self.assertEqual(logits.grad[:, :, :2].count_nonzero().item(), 0)

    def test_auxiliary_is_not_boundary_weighted(self) -> None:
        targets = torch.randint(0, 2, (2, 16, 16))
        main = torch.randn(2, 1, 16, 16)
        aux = torch.randn(2, 1, 4, 4)
        criterion = BoILLoss(grid_sizes=(8,), aux_weight=0.1)
        increments = []
        for epoch in (0, 3):
            increments.append(
                criterion({"main": main, "aux": aux}, targets, epoch)
                - criterion({"main": main}, targets, epoch)
            )
        torch.testing.assert_close(increments[0], increments[1])

    def test_all_ignore_is_finite_zero(self) -> None:
        logits = torch.randn(2, 3, 16, 16, requires_grad=True)
        criterion = BoILLoss(grid_sizes=(8,))
        targets = torch.full((2, 16, 16), 255)
        loss = criterion({"main": logits}, targets, 3)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(logits.grad.count_nonzero().item(), 0)

    def test_zero_warmup_matches_region_loss(self) -> None:
        criterion = BoILLoss(grid_sizes=(8,))
        logits = torch.randn(2, 1, 16, 16)
        targets = torch.randint(0, 2, (2, 16, 16))
        torch.testing.assert_close(
            criterion({"main": logits}, targets, epoch=0), criterion.segmentation_loss(logits, targets)
        )

    def test_encoder_rejects_bad_shapes_and_tokens(self) -> None:
        encoder = FrozenDINOv3(ToyBackbone(), (1, 2))
        with self.assertRaises(ValueError):
            encoder(torch.rand(1, 3, 33, 32))
        with self.assertRaises(ValueError):
            encoder(torch.full((1, 3, 32, 32), 255.0))
        original_forward = encoder.backbone.forward

        def malformed(**kwargs):
            output = original_forward(**kwargs)
            return SimpleNamespace(hidden_states=tuple(x[:, :-1] for x in output.hidden_states))

        encoder.backbone.forward = malformed
        with self.assertRaises(ValueError):
            encoder(torch.rand(1, 3, 32, 32))

    def test_prediction_with_decoder_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads(Path("configs/smoke.json").read_text(encoding="utf-8"))
            config["encoder_path"] = "mock-checkpoint"
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            image_path = root / "image.png"
            Image.new("RGB", (48, 80), (50, 120, 20)).save(image_path)
            for classes in (1, 3):
                config["model"]["num_classes"] = classes
                config_path.write_text(json.dumps(config), encoding="utf-8")
                model = self.make_model(classes)
                state = {k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}
                checkpoint = root / "decoder.pt"
                torch.save(state, checkpoint)
                output = root / f"mask_{classes}.png"
                argv = [
                    "predict", "--config", str(config_path), "--checkpoint", str(checkpoint),
                    "--image", str(image_path), "--output", str(output),
                ]
                encoder = FrozenDINOv3(ToyBackbone(), (1, 2))
                with patch("sys.argv", argv), patch.object(FrozenDINOv3, "from_pretrained", return_value=encoder):
                    with contextlib.redirect_stdout(io.StringIO()):
                        predict()
                with Image.open(output) as mask:
                    self.assertEqual(mask.size, (48, 80))
                    self.assertEqual(mask.mode, "L")
                    values = set(mask.getdata())
                    self.assertTrue(values <= ({0, 255} if classes == 1 else {0, 1, 2}))


if __name__ == "__main__":
    unittest.main()
