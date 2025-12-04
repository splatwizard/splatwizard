from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from splatwizard.utils.graphics_utils import getWorld2View2, getProjectionMatrix
from torch import Tensor


@dataclass
class Camera:
    uid: int
    colmap_id:int

    R: Tensor
    T: Tensor
    FoVx: float
    FoVy: float
    image_name: str
    scale_mat: Tensor = None
    world_mat: Tensor = None
    data_device: str = 'cuda'

    image: Tensor = None

    orig_w: int = None
    orig_h: int = None

    original_image: Tensor = None
    image_width: int = None
    image_height: int = None
    gt_alpha_mask: Tensor = None

    zfar: float = 100.0
    znear: float = 0.01

    trans: Tensor = np.array([0.0, 0.0, 0.0])
    scale: float = 1.0

    world_view_transform: Tensor = None
    projection_matrix: Tensor = None
    full_proj_transform: Tensor = None
    camera_center: Tensor = None

    focal_x: float = None
    focal_y: float = None
    position: Tensor = None
    rotation: Tensor = None

    info: Any = None

    def __post_init__(self):

        try:
            self.data_device = torch.device(self.data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {self.data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        if self.image is not None:
            self.set_image(self.image, self.gt_alpha_mask)

        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def set_image(self, image, gt_alpha_mask=None, mask_background=True):
        self.original_image = image.to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            if mask_background:
                self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            # self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.gt_alpha_mask = None



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

