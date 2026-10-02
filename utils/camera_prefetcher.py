"""Bounded, lossless camera prefetch with explicit CUDA stream ownership."""

from dataclasses import dataclass, field
import queue
from random import randint
import threading

import torch


@dataclass
class PrefetchedCamera:
    camera: object
    rgb: torch.Tensor
    alpha: torch.Tensor
    ready: object = None
    sources: tuple = field(default_factory=tuple)

    def __enter__(self):
        if self.ready is not None:
            stream = torch.cuda.current_stream(self.rgb.device)
            stream.wait_event(self.ready)
            self.rgb.record_stream(stream)
            self.alpha.record_stream(stream)
        return self

    def close(self):
        # Host sources must stay alive until any asynchronous copy completes.
        if self.ready is not None:
            self.ready.synchronize()
        self.rgb = self.alpha = self.ready = None
        self.sources = ()

    def __exit__(self, *args):
        self.close()


class CameraPrefetcher:
    """Sample without replacement and prepare at most queue_size + 1 frames.

    The producer retries the same frame when the queue is full. Decode/upload
    errors reach the consumer, and shutdown releases pending and queued frames.
    """

    def __init__(self, cameras, num_batches=None, queue_size=2, device="cuda"):
        self._cameras = list(cameras)
        if not self._cameras:
            raise ValueError("Cannot prefetch an empty training set")
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        if num_batches is not None and num_batches < 0:
            raise ValueError("num_batches must be nonnegative")
        self._num_batches = num_batches
        self._stack = []
        self._device = torch.device(device)
        if self._device.type == "cuda" and self._device.index is None:
            self._device = torch.device("cuda", torch.cuda.current_device())
        self._queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._finished = threading.Event()
        self._error = None
        self._worker = threading.Thread(target=self._run, name="camera-prefetch", daemon=True)
        self._worker.start()

    def _pick(self):
        if not self._stack:
            self._stack = self._cameras.copy()
        return self._stack.pop(randint(0, len(self._stack) - 1))

    def _prepare(self, camera, stream):
        ready = None
        try:
            if stream is None:
                sources = camera.image_tensors()
                rgb, alpha = (tensor.to(self._device) for tensor in sources)
            else:
                with torch.cuda.stream(stream):
                    sources = camera.image_tensors()
                    rgb, alpha = (tensor.to(self._device, non_blocking=True) for tensor in sources)
                    ready = torch.cuda.Event()
                    ready.record(stream)
            return PrefetchedCamera(camera, rgb, alpha, ready, sources)
        finally:
            camera.release_loaded()

    def _run(self):
        pending = None
        try:
            if self._device.type == "cuda":
                torch.cuda.set_device(self._device)
                stream = torch.cuda.Stream(device=self._device)
            else:
                stream = None
            produced = 0
            while not self._stop.is_set() and (self._num_batches is None or produced < self._num_batches):
                pending = self._prepare(self._pick(), stream)
                while not self._stop.is_set():
                    try:
                        self._queue.put(pending, timeout=0.1)
                        pending = None
                        produced += 1
                        break
                    except queue.Full:
                        continue
                if pending is not None:
                    pending.close()
                    pending = None
        except Exception as error:
            self._error = error
        finally:
            if pending is not None:
                pending.close()
            self._finished.set()

    def next(self):
        while not self._stop.is_set():
            try:
                return self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._error is not None:
                    raise RuntimeError("Camera prefetch failed") from self._error
                if self._finished.is_set():
                    raise StopIteration
        raise StopIteration

    def shutdown(self):
        self._stop.set()
        # Joining first also prevents the producer from putting a frame after
        # the queue has been drained. Queue puts have a bounded timeout.
        self._worker.join(timeout=10)
        if self._worker.is_alive():
            raise RuntimeError("Camera prefetch worker did not stop")
        while True:
            try:
                self._queue.get_nowait().close()
            except queue.Empty:
                break

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()
