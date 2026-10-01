"""CUDA integration checks for conditional appearance, topology, and persistence."""

import json
from pathlib import Path
import tempfile
import unittest
from argparse import ArgumentParser

import numpy as np
from PIL import Image
import torch

from arguments import OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene.cameras import Camera
from scene.dataset_readers import readCamerasFromTransforms
from scene.gaussian_model import GaussianModel
from scene.sun_conditioning import sun_sh_encoding
from utils.graphics_utils import BasicPointCloud
from utils.light_utils import normalized_sun_direction
from utils.sh_utils import eval_sh
from tools.eval_test_groups import group_metrics


class EvaluationGroupingTests(unittest.TestCase):
    def make_run(self, root):
        source = root / "dataset"
        source.mkdir()
        training = [{"file_path": "cam00/0001.png", "time_index": 1},
                    {"file_path": "cam01/0002.png", "time_index": 2}]
        testing = [{"file_path": "cam02/0001.png", "time_index": 1},
                   {"file_path": "cam00/0007.png", "time_index": 7},
                   {"file_path": "cam01/0007.png", "time_index": 7}]
        for split, frames in (("train", training), ("test", testing)):
            (source / f"transforms_{split}.json").write_text(json.dumps({"frames": frames}))
        (root / "training_config.json").write_text(json.dumps({
            "model": {"eval": True, "source_path": str(source)}}))
        method = root / "test" / "ours_1000"
        method.mkdir(parents=True)
        manifest = [{"image": f"{index:05d}.png", **frame} for index, frame in enumerate(testing)]
        (method / "manifest.json").write_text(json.dumps(manifest))
        (root / "per_view.json").write_text(json.dumps({"ours_1000": {
            "PSNR": {entry["image"]: score for entry, score in zip(manifest, (30., 20., 24.))}}}))
        return method

    def test_group_metrics_follow_actual_heldout_suns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_run(root)
            result = group_metrics(root)
            self.assertEqual(result["heldout_sun_ids"], [7])
            groups = result["results"]["ours_1000"]
            self.assertEqual(groups["heldout_sun"], {"n": 2, "PSNR": 22.})
            self.assertEqual(groups["seen_sun_new_combination"], {"n": 1, "PSNR": 30.})
            self.assertAlmostEqual(groups["all"]["PSNR"], 74. / 3)

    def test_group_metrics_reject_missing_or_mislabelled_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            method = self.make_run(root)
            path = method / "manifest.json"
            manifest = json.loads(path.read_text())
            path.write_text(json.dumps(manifest[:-1]))
            with self.assertRaisesRegex(ValueError, "full test split"):
                group_metrics(root)
            manifest[0]["time_index"] = 7
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "sun ID"):
                group_metrics(root)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA integration tests require a GPU")
class SunConditioningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        rng = np.random.default_rng(42)
        self.camera = Camera((64, 64), 0, np.eye(3), np.array([0., 0., 4.]),
                             1.0, 1.0, None, Image.new("RGB", (64, 64)), None,
                             "cam00/images/0000.png", 0, data_device="cpu",
                             sun_direction=[0, 1, 0])
        self.pcd = BasicPointCloud(rng.uniform(-0.4, 0.4, (24, 3)).astype(np.float32),
                                   np.full((24, 3), 0.65, np.float32),
                                   np.zeros((24, 3), np.float32))
        parser = ArgumentParser()
        self.options = OptimizationParams(parser).extract(parser.parse_args([]))
        parser = ArgumentParser()
        self.pipe = PipelineParams(parser).extract(parser.parse_args([]))
        self.bg = torch.zeros(3, device="cuda")

    def model(self):
        model = GaussianModel(3)
        model.create_from_pcd(self.pcd, [self.camera], 1.)
        model.training_setup(self.options)
        model.active_sh_degree = 3
        with torch.no_grad():
            model._features_rest.normal_(std=0.01)
        return model

    def image(self, model, direction=None, **kwargs):
        return render(self.camera, model, self.pipe, self.bg,
                      sun_direction=direction, **kwargs)["render"]

    def update(self, model):
        loss = (self.image(model) - 0.15).square().mean()
        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        model.sun_optimizer.step()
        model.sun_optimizer.zero_grad(set_to_none=True)
        return loss.item()

    def test_zero_residual_matches_cuda_images_and_geometry_gradients(self):
        model = self.model()
        weights = torch.linspace(.2, 1.0, 3 * 64 * 64, device="cuda").reshape(3, 64, 64)
        model.sun_conditioning = False
        reference = self.image(model)
        (reference * weights).mean().backward()
        parameters = [model._xyz, model._features_dc, model._features_rest,
                      model._opacity, model._scaling, model._rotation]
        gradients = [parameter.grad.clone() for parameter in parameters]
        for parameter in parameters:
            parameter.grad = None
        model.sun_conditioning = True
        conditional = self.image(model, separate_sh=True)
        torch.testing.assert_close(conditional, reference, atol=2e-6, rtol=2e-5)
        (conditional * weights).mean().backward()
        for parameter, reference_grad in zip(parameters, gradients):
            torch.testing.assert_close(parameter.grad, reference_grad, atol=2e-6, rtol=2e-4)

    def test_sun_encoding_and_input_validation(self):
        direction = torch.tensor([.3, .7, -.2], device="cuda")
        direction /= direction.norm()
        expected = eval_sh(2, torch.eye(9, device="cuda"), direction)
        torch.testing.assert_close(sun_sh_encoding(direction), expected)
        for invalid in ([0, 0, 0], [0, float("nan"), 1], [1, 2]):
            with self.assertRaises(ValueError):
                normalized_sun_direction(invalid)

    def test_direction_changes_image_and_all_conditioning_gradients_flow(self):
        model = self.model()
        with torch.no_grad():
            model.sun_decoder.layers[-1].weight.normal_(std=.02)
        first = self.image(model, [0, 1, 0])
        second = self.image(model, [1, 1, 0])
        self.assertGreater((first - second).abs().max().item(), 1e-6)
        (first - .15).square().mean().backward()
        self.assertTrue(torch.isfinite(model._sun_features.grad).all())
        self.assertGreater(model._sun_features.grad.abs().sum().item(), 0)
        for parameter in model.sun_decoder.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_clone_split_prune_preserve_features_and_optimizer_shapes(self):
        model = self.model()
        model._sun_features.sum().backward()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        decoder_ids = [id(parameter) for parameter in model.sun_decoder.parameters()]
        with torch.no_grad():
            model._scaling.fill_(np.log(.001))
        model.tmp_radii = torch.ones(24, device="cuda")
        gradients = torch.zeros((24, 1), device="cuda")
        gradients[2] = 1
        parent = model._sun_features[2].detach().clone()
        model.densify_and_clone(gradients, .5, 1.)
        self.assertEqual(model._xyz.shape[0], 25)
        torch.testing.assert_close(model._sun_features[-1], parent)
        with torch.no_grad():
            model._scaling[1].fill_(np.log(.1))
        gradients = torch.zeros((25, 1), device="cuda")
        gradients[1] = 1
        parent = model._sun_features[1].detach().clone()
        model.densify_and_split(gradients, .5, 1.)
        self.assertEqual(model._xyz.shape[0], 26)
        torch.testing.assert_close(model._sun_features[-2:], parent.expand(2, -1))
        before = model._sun_features.detach().clone()
        prune = torch.arange(26, device="cuda") % 3 == 0
        model.prune_points(prune)
        torch.testing.assert_close(model._sun_features, before[~prune])
        for group in model.optimizer.param_groups:
            parameter = group["params"][0]
            self.assertEqual(parameter.shape[0], model._xyz.shape[0])
            state = model.optimizer.state.get(parameter)
            if state:
                self.assertEqual(state["exp_avg"].shape, parameter.shape)
                self.assertEqual(state["exp_avg_sq"].shape, parameter.shape)
        self.assertEqual(decoder_ids, [id(parameter) for parameter in model.sun_decoder.parameters()])

    def test_ply_sidecar_and_baked_ply_render_consistently(self):
        model = self.model()
        self.update(model)
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            source = Path(directory) / "point_cloud.ply"
            model.save_ply(str(source))
            loaded = GaussianModel(3)
            loaded.load_ply(str(source))
            torch.testing.assert_close(self.image(loaded), self.image(model), atol=1e-7, rtol=1e-6)
            baked = Path(directory) / "baked.ply"
            model.save_ply(str(baked), sun_direction=[1, 1, 0])
            ordinary = GaussianModel(3)
            ordinary.load_ply(str(baked))
            self.assertFalse(ordinary.sun_conditioning)
            torch.testing.assert_close(self.image(ordinary), self.image(model, [1, 1, 0]),
                                       atol=2e-6, rtol=2e-5)
            source.with_name("sun_conditioning.pt").unlink()
            with self.assertRaises(FileNotFoundError):
                loaded.load_ply(str(source))

    def test_serialized_checkpoint_restores_optimizer_and_next_update(self):
        model = self.model()
        self.update(model)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            torch.save(model.capture(), checkpoint)
            restored = self.model()
            restored.restore(torch.load(checkpoint, map_location="cuda", weights_only=False), self.options)
        self.assertEqual(restored.exposure_mapping, model.exposure_mapping)
        torch.testing.assert_close(self.image(restored), self.image(model), atol=1e-7, rtol=1e-6)
        self.update(model)
        self.update(restored)
        torch.testing.assert_close(model._xyz, restored._xyz)
        torch.testing.assert_close(model._sun_features, restored._sun_features)
        for left, right in zip(model.sun_decoder.parameters(), restored.sun_decoder.parameters()):
            torch.testing.assert_close(left, right)

    def test_repeated_image_stems_keep_unique_ids_and_sun_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = []
            for camera in (0, 1):
                relative = f"cam{camera:02d}/images/0000.png"
                (root / relative).parent.mkdir(parents=True)
                Image.new("RGB", (16, 16)).save(root / relative)
                frames.append({"file_path": relative, "transform_matrix": np.eye(4).tolist(),
                               "sun_direction": [0, 2, 0], "camera_index": camera, "time_index": 0})
            (root / "transforms_train.json").write_text(json.dumps({
                "camera_angle_x": 1.0, "frames": frames}), encoding="utf-8")
            loaded = readCamerasFromTransforms(str(root), "transforms_train.json", "", False, False)
            self.assertEqual(len({camera.image_name for camera in loaded}), 2)
            np.testing.assert_allclose(loaded[0].sun_direction, [0, 1, 0])
            self.assertEqual(loaded[1].camera_index, 1)


if __name__ == "__main__":
    unittest.main()
