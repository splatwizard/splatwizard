import os
import pathlib
import random

from PIL import Image
from loguru import logger
import torch
from torch.utils.data import Dataset
import numpy as np

from splatwizard._cmod.lanczos_resampling import lanczos_resample
from splatwizard.config import PipelineParams, EvalMode
from splatwizard.utils.general_utils import PILtoTorch

WARNED = False

class ViewDataset(Dataset):
    def __init__(self, cameras, ppl: PipelineParams, require_image=True):
        self.cameras = cameras
        self.ppl = ppl
        # self.resolution_scales = resolution_scales

        # self._height = tensor_image.shape[-2]
        # self._width = tensor_image.shape[-1]
        self.require_image = require_image
        self.cache = {}
        self.cached_shape = {}

    def __len__(self):
        return len(self.cameras)

    def calculate_resolution(self, img, resolution_scale):
        orig_w, orig_h = img.size
        if self.ppl.resolution in [1, 2, 4, 8]:
            resolution = round(orig_w / (resolution_scale * self.ppl.resolution)), round(
                orig_h / (resolution_scale * self.ppl.resolution))
        else:  # should be a type that converts to float
            if self.ppl.resolution == -1:
                if orig_w > 1600:
                    # global WARNED
                    # if not WARNED:
                    #     # print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                    #     #     "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    #     logger.info("Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.")
                    #     logger.info("If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    #     WARNED = True
                    global_down = orig_w / 1600
                else:
                    global_down = 1
            else:
                global_down = orig_w / self.ppl.resolution

            scale = float(global_down) * float(resolution_scale)
            resolution = (int(orig_w / scale), int(orig_h / scale))

        return resolution

    def load_cam(self, cam_info, image, resolution):
        if self.ppl.lanczos_resample:
            image = torch.from_numpy(np.array(image)).cuda() / 255.0
            resized_image_rgb = lanczos_resample(image, size=resolution).permute(2, 1, 0).cpu()
        else:
            resized_image_rgb = PILtoTorch(image, resolution)
        # resized_image_rgb = PILtoTorch(image, resolution)

        gt_image = resized_image_rgb[:3, ...]
        loaded_mask = torch.Tensor([])

        # print(f'gt_image: {gt_image.shape}')
        if resized_image_rgb.shape[0] == 4:
            loaded_mask = resized_image_rgb[3, ...]



        return cam_info.ex_id, gt_image, loaded_mask

    def fetch_data(self, idx):
        cam_info = self.cameras[idx]
        results = []
        # print('open', cam_info.image_path)
        image = Image.open(cam_info.image_path)
        resolution = self.calculate_resolution(image, 1.0)
        return self.load_cam(cam_info, image, resolution), (image.size[0], image.size[1])



    def __getitem__(self, idx):
        if self.ppl.cache_dataset:
            try:
                data_pack = self.cache[idx]
            except KeyError:
                self.cache[idx] = self.fetch_data(idx)
                data_pack = self.cache[idx]
        else:
            data_pack = self.fetch_data(idx)


        (ex_id, gt_image, loaded_mask), (orig_w, orig_h) = data_pack
        if self.require_image:
            return ex_id, gt_image, loaded_mask, gt_image.shape, (orig_w, orig_h)
        else:
            return ex_id, torch.Tensor(), torch.Tensor(), gt_image.shape, (orig_w, orig_h)
