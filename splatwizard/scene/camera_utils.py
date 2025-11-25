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

from splatwizard.config import PipelineParams

from .cameras import Camera
import numpy as np
from splatwizard.utils.general_utils import PILtoTorch
from splatwizard.utils.graphics_utils import fov2focal
from loguru import logger

from torchvision.transforms import ToTensor
from .._cmod.lanczos_resampling import lanczos_resample
from ..profiler import profile

WARNED = False

def loadCam(ppl: PipelineParams, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if ppl.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * ppl.resolution)), round(orig_h / (resolution_scale * ppl.resolution))
    else:  # should be a type that converts to float
        if ppl.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    # print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                    #     "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    logger.info("Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.")
                    logger.info("If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / ppl.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    if ppl.lanczos_resample:
        image = torch.from_numpy(np.array(cam_info.image)).cuda() / 255.0
        resized_image_rgb = lanczos_resample(image, size=resolution).permute(2, 1, 0)
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)


    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    # print(f'gt_image: {gt_image.shape}')
    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    cam = Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                 FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                 image=gt_image, gt_alpha_mask=loaded_mask,
                 image_name=cam_info.image_name, uid=id, data_device=ppl.data_device)

    cam.info = cam_info

    cam.width = cam.image_width
    cam.height = cam.image_height
    cam.FovX = cam.FoVx
    cam.FovY = cam.FoVy
    cam_json = camera_to_JSON(0, cam)

    cam.focal_x = cam_json['fx']
    cam.focal_y = cam_json['fy']
    cam.position = torch.tensor(cam_json['position'])
    cam.rotation = torch.tensor(cam_json['rotation'])

    return cam

def cameraList_from_camInfos(cam_infos, resolution_scale, ppl):
    camera_list = []

    if ppl.debug:
        for id, c in enumerate(cam_infos):
            camera_list.append(loadCam(ppl, id, c, resolution_scale))
            if id > 5:
                break
    else:
        for id, c in enumerate(cam_infos):
            camera_list.append(loadCam(ppl, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


