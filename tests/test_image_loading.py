"""Regression checks for GT values, bounded ownership and prefetch failures."""

import gc
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import weakref

import numpy as np
from PIL import Image
import torch

from utils.camera_prefetcher import CameraPrefetcher
from utils.general_utils import PILtoTorch
from utils.image_cache import DecodedImageCache, decoded_image_cache


class DecodedCacheTests(unittest.TestCase):
    def test_preserves_pil_pixels_alpha_and_resize(self):
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as directory:
            cache = DecodedImageCache()
            for channels in (1, 3, 4):
                shape = (19, 23) if channels == 1 else (19, 23, channels)
                image = Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8))
                path = Path(directory) / f"channels_{channels}.png"
                image.save(path)
                for resolution in ((23, 19), (11, 9), (31, 27)):
                    pixels = cache.get(path, resolution)
                    self.assertEqual(pixels.dtype, torch.uint8)
                    actual = pixels / 255.0
                    if actual.ndim == 2:
                        actual = actual.unsqueeze(-1)
                    torch.testing.assert_close(actual.permute(2, 0, 1), PILtoTorch(image, resolution),
                                               rtol=0, atol=0)

    def test_lru_limits_payload_and_reuses_decoded_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = DecodedImageCache()
            paths = [Path(directory) / f"{index}.png" for index in range(3)]
            for path in paths:
                Image.new("RGBA", (16, 12)).save(path)
            with patch("utils.image_cache.Image.open", wraps=Image.open) as opened:
                cache.get(paths[0], (16, 12), 2)
                cache.get(paths[1], (16, 12), 2)
                cache.get(paths[0], (16, 12), 2)
                self.assertEqual(opened.call_count, 2)
                cache.get(paths[2], (16, 12), 2)
                self.assertEqual(cache.stats(), {"entries": 2, "bytes": 2 * 16 * 12 * 4})
                cache.get(paths[1], (16, 12), 2)
                self.assertEqual(opened.call_count, 4)
            cache.clear()
            self.assertEqual(cache.stats(), {"entries": 0, "bytes": 0})


class StubCamera:
    def __init__(self, index, prepared=None, references=None, error=False):
        self.index = index
        self.prepared = prepared
        self.references = references
        self.error = error
        self.loaded = None

    def image_tensors(self):
        if self.error:
            raise FileNotFoundError("missing training PNG")
        self.loaded = (torch.full((3, 8, 8), self.index / 10.), torch.ones((1, 8, 8)))
        if self.references is not None:
            self.references.extend(weakref.ref(tensor) for tensor in self.loaded)
        if self.prepared is not None:
            self.prepared.set()
        return self.loaded

    def release_loaded(self):
        self.loaded = None


class PrefetchTests(unittest.TestCase):
    def test_full_queue_keeps_sampling_without_replacement(self):
        # The producer must keep the second frame while waiting for this
        # deliberately slow consumer, rather than drawing and losing it.
        prepared = threading.Event()
        cameras = [StubCamera(index, prepared) for index in range(5)]
        with CameraPrefetcher(cameras, num_batches=10, queue_size=1, device="cpu") as prefetcher:
            self.assertTrue(prepared.wait(3))
            # Wait long enough to exercise at least one queue-put timeout.
            threading.Event().wait(0.25)
            indices = []
            for _ in range(10):
                with prefetcher.next() as batch:
                    indices.append(batch.camera.index)
                    self.assertIsNone(batch.camera.loaded)
            self.assertEqual(sorted(indices[:5]), list(range(5)))
            self.assertEqual(sorted(indices[5:]), list(range(5)))
            with self.assertRaises(StopIteration):
                prefetcher.next()
        self.assertFalse(prefetcher._worker.is_alive())

    def test_shutdown_releases_pending_and_queued_tensors(self):
        references = []
        prepared = threading.Event()
        cameras = [StubCamera(index, prepared, references) for index in range(3)]
        prefetcher = CameraPrefetcher(cameras, queue_size=1, device="cpu")
        self.assertTrue(prepared.wait(3))
        threading.Event().wait(0.25)
        prefetcher.shutdown()
        gc.collect()
        self.assertFalse(prefetcher._worker.is_alive())
        self.assertTrue(all(camera.loaded is None for camera in cameras))
        self.assertTrue(references)
        self.assertTrue(all(reference() is None for reference in references))

    def test_decode_failure_reaches_consumer_and_stops_worker(self):
        prefetcher = CameraPrefetcher([StubCamera(0, error=True)], device="cpu")
        try:
            with self.assertRaisesRegex(RuntimeError, "Camera prefetch failed") as raised:
                prefetcher.next()
            self.assertIsInstance(raised.exception.__cause__, FileNotFoundError)
        finally:
            prefetcher.shutdown()
        self.assertFalse(prefetcher._worker.is_alive())


@unittest.skipUnless(torch.cuda.is_available(), "Cameras require CUDA pose matrices")
class LazyCameraTests(unittest.TestCase):
    def setUp(self):
        # Warm the inverse wrapper before constructing cameras, as loadCam does.
        torch.eye(4, device="cuda").inverse()
        decoded_image_cache.clear()

    def tearDown(self):
        decoded_image_cache.clear()

    def camera(self, path, resolution, **kwargs):
        from scene.cameras import Camera
        return Camera(resolution, 0, np.eye(3), np.array([0., 0., 4.]),
                      1., 1., None, None, None, "cam00/images/0000.png", 0,
                      image_path=path, data_device="cpu", sun_direction=[0, 1, 0], **kwargs)

    def test_file_camera_is_lazy_and_float_release_preserves_uint8_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gt.png"
            image = Image.fromarray(np.random.default_rng(5).integers(0, 256, (19, 23, 4), dtype=np.uint8))
            image.save(path)
            camera = self.camera(path, (11, 9))
            self.assertEqual(decoded_image_cache.stats()["entries"], 0)
            self.assertIsNone(camera._loaded_rgb)
            eager = PILtoTorch(image, (11, 9))
            with patch("utils.image_cache.Image.open", wraps=Image.open) as opened:
                torch.testing.assert_close(camera.original_image, eager[:3].clamp(0, 1), rtol=0, atol=0)
                torch.testing.assert_close(camera.alpha_mask, eager[3:4], rtol=0, atol=0)
                camera.release_loaded()
                self.assertIsNone(camera._loaded_rgb)
                self.assertIsNone(camera._loaded_alpha)
                self.assertEqual(decoded_image_cache.stats(), {"entries": 1, "bytes": 11 * 9 * 4})
                torch.testing.assert_close(camera.original_image, eager[:3], rtol=0, atol=0)
                self.assertEqual(opened.call_count, 1)
                camera.release_loaded()

    def test_exposure_half_mask_does_not_modify_cached_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gt.png"
            Image.new("RGBA", (12, 8), (30, 60, 90, 128)).save(path)
            for is_test_dataset in (False, True):
                camera = self.camera(path, (12, 8), train_test_exp=True,
                                     is_test_view=True, is_test_dataset=is_test_dataset)
                expected = torch.full((1, 8, 12), 128 / 255.)
                if is_test_dataset:
                    expected[..., :6] = 0
                else:
                    expected[..., 6:] = 0
                torch.testing.assert_close(camera.alpha_mask, expected, rtol=0, atol=0)
                camera.release_loaded()
            camera = self.camera(path, (12, 8))
            torch.testing.assert_close(camera.alpha_mask, torch.full((1, 8, 12), 128 / 255.), rtol=0, atol=0)
            camera.release_loaded()

    def test_cuda_prefetch_matches_eager_gt_loss_and_gradient(self):
        from fused_ssim import fused_ssim
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gt.png"
            image = Image.fromarray(np.random.default_rng(5).integers(0, 256, (64, 64, 4), dtype=np.uint8))
            image.save(path)
            eager = PILtoTorch(image, (64, 64)).cuda()
            camera = self.camera(path, (64, 64))
            def loss_and_gradient(rgb, alpha):
                prediction = torch.full_like(rgb, 0.5, requires_grad=True)
                masked = prediction * alpha
                loss = 0.8 * (masked - rgb).abs().mean() + 0.2 * (1 - fused_ssim(masked[None], rgb[None]))
                loss.backward()
                return loss.detach(), prediction.grad
            expected = loss_and_gradient(eager[:3], eager[3:4])
            with CameraPrefetcher([camera], num_batches=1) as prefetcher:
                with prefetcher.next() as batch:
                    torch.testing.assert_close(batch.rgb, eager[:3], rtol=0, atol=0)
                    torch.testing.assert_close(batch.alpha, eager[3:4], rtol=0, atol=0)
                    actual = loss_and_gradient(batch.rgb, batch.alpha)
                    for left, right in zip(actual, expected):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
            self.assertIsNone(camera._loaded_rgb)


if __name__ == "__main__":
    unittest.main()
