"""CPU image cache shared by lazy cameras; pixels retain PIL resize semantics."""

from collections import OrderedDict
import os
import threading

import numpy as np
from PIL import Image
import torch


def pil_to_cached_tensor(image, resolution):
    # Match PILtoTorch's resize, channel order and input dtype. In particular,
    # keep RGBA rather than converting to RGB and silently discarding alpha.
    return torch.from_numpy(np.array(image.resize(resolution)))


class DecodedImageCache:
    def __init__(self):
        self._images = OrderedDict()
        self._lock = threading.Lock()

    def get(self, path, resolution, max_entries=0):
        if max_entries < 0:
            raise ValueError("image_cache_max must be nonnegative")
        key = (os.path.abspath(os.fspath(path)), tuple(resolution))
        with self._lock:
            if key in self._images:
                self._images.move_to_end(key)
                self._trim(max_entries)
                return self._images[key]

        with Image.open(path) as image:
            decoded = pil_to_cached_tensor(image, resolution)
        with self._lock:
            # Another reader may have filled this key while we were decoding.
            decoded = self._images.setdefault(key, decoded)
            self._images.move_to_end(key)
            self._trim(max_entries)
            return decoded

    def _trim(self, max_entries):
        while max_entries and len(self._images) > max_entries:
            self._images.popitem(last=False)

    def clear(self):
        with self._lock:
            self._images.clear()

    def stats(self):
        with self._lock:
            return {"entries": len(self._images),
                    "bytes": sum(image.numel() * image.element_size()
                                 for image in self._images.values())}


decoded_image_cache = DecodedImageCache()


def memory_usage():
    """Optional Linux diagnostics; cgroup quota is separate from host RAM."""
    result = {}
    try:
        with open("/proc/self/status") as stream:
            for line in stream:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    key = "process_rss_bytes" if line.startswith("VmRSS:") else "process_peak_rss_bytes"
                    result[key] = int(line.split()[1]) * 1024
        with open("/sys/fs/cgroup/memory.current") as stream:
            result["container_memory_bytes"] = int(stream.read())
    except (OSError, ValueError):
        pass
    return result
