#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.image_cache import decoded_image_cache, pil_to_cached_tensor
from utils.light_utils import normalized_sun_direction
import cv2
import threading

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False,
                 sun_direction=None, camera_index=None, time_index=None,
                 image_path=None, image_cache_max=0
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.camera_index = camera_index
        self.time_index = time_index
        self.v_l = (torch.as_tensor(normalized_sun_direction(sun_direction),
                                    device="cuda", dtype=torch.float32)
                    if sun_direction is not None else None)

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.image_path = image_path
        self.image_cache_max = image_cache_max
        self._resolution = tuple(resolution)
        self.image_width, self.image_height = self._resolution
        if image is None and image_path is None:
            raise ValueError("Camera requires an image or image_path")
        # Direct PIL input remains supported for small programmatic cameras.
        # Dataset cameras carry only a path until they are sampled.
        self._image_pixels = pil_to_cached_tensor(image, resolution) if image is not None else None
        self._loaded_rgb = None
        self._loaded_alpha = None
        self._image_lock = threading.Lock()
        self._train_test_exp = train_test_exp
        self._is_test_dataset = is_test_dataset
        self._is_test_view = is_test_view

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones((1, self.image_height, self.image_width),
                                         device=self.data_device)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def image_tensors(self):
        """Return one consistent RGB/alpha pair, even if another reader releases it."""
        with self._image_lock:
            if self._loaded_rgb is None:
                pixels = self._image_pixels
                if pixels is None:
                    pixels = decoded_image_cache.get(self.image_path, self._resolution,
                                                     self.image_cache_max)
                # Normalize on CPU exactly as the previous PILtoTorch did.
                # Move only this frame's floats to data_device, not the whole set.
                floats = pixels / 255.0
                if floats.ndim == 2:
                    floats = floats.unsqueeze(-1)
                floats = floats.permute(2, 0, 1)
                rgb = floats[:3].clamp(0.0, 1.0).to(self.data_device)
                alpha = (floats[3:4].to(self.data_device) if floats.shape[0] == 4
                         else torch.ones((1, self.image_height, self.image_width),
                                         device=self.data_device))
                if self._train_test_exp and self._is_test_view:
                    midpoint = self.image_width // 2
                    if self._is_test_dataset:
                        alpha[..., :midpoint] = 0
                    else:
                        alpha[..., midpoint:] = 0
                self._loaded_rgb, self._loaded_alpha = rgb, alpha
            return self._loaded_rgb, self._loaded_alpha

    @property
    def original_image(self):
        return self.image_tensors()[0]

    @property
    def alpha_mask(self):
        return self.image_tensors()[1]

    def release_loaded(self):
        """Drop this camera's float buffers; retain only the shared CPU pixel cache."""
        with self._image_lock:
            self._loaded_rgb = self._loaded_alpha = None
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
