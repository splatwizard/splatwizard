import copy
import os
import pathlib


import torch.utils.data
from PIL import Image


from splatwizard.data_loader.dataset_readers import readColmapSceneInfo, readNerfSyntheticInfo
from splatwizard.config import PipelineParams, EvalMode, DataMode, OptimizationParams
from splatwizard.scene.camera_utils import cameraList_from_camInfos, camera_to_JSON
from loguru import logger

from splatwizard.scene.cameras import Camera
from splatwizard.scene.dataset import ViewDataset


class CameraIterator:
    def __init__(self, cameras, ppl: PipelineParams, shuffle=False, batch_size=1, require_image=True):
        # logger.info(f'Shuffled data {shuffle}')
        assert batch_size == 1
        self.cameras_info_dict = {c.ex_id: c for c in cameras}
        self.cameras_dict = {}
        self.ppl = ppl
        self.require_image = require_image

        if self.ppl.cache_dataset and self.ppl.num_workers != 0:
            logger.warning('Using cache_dataset and multiprocessing dataloader simultaneously is not recommended')

        self.view_dataset = ViewDataset(cameras, ppl=ppl, require_image=require_image)
        self.dataloader = torch.utils.data.DataLoader(
            self.view_dataset, batch_size=batch_size,
            shuffle=shuffle,
            num_workers=ppl.num_workers, pin_memory=True,
            drop_last=False
        )

        self.pre_build_cameras()
        self.enumerator = enumerate(self.dataloader)

        self._shuffle = shuffle

    @property
    def shuffle(self):
        return self._shuffle



    def pre_build_cameras(self):
        for cam_id, cam_info in self.cameras_info_dict.items():
            cam = Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                         FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                         image=None, gt_alpha_mask=None,
                         image_name=cam_info.image_name, uid=None,
                         scale_mat=cam_info.scale_mat, world_mat=cam_info.world_mat,
                         data_device=self.ppl.data_device)

            cam.info = cam_info

            cam.width = cam.image_width
            cam.height = cam.image_height
            cam.FovX = cam.FoVx
            cam.FovY = cam.FoVy



            self.cameras_dict[cam_id] = cam

    def build_camera(self, cam_id, gt_image, loaded_mask=None):
        cam = copy.deepcopy(self.cameras_dict[cam_id])
        # print(cam.image_name, cam.R)
        cam.set_image(image=gt_image, gt_alpha_mask=loaded_mask, mask_background=self.ppl.mask_background)

        cam.width = cam.image_width
        cam.height = cam.image_height

        # loadCam(
        cam_json = camera_to_JSON(0, cam)

        cam.focal_x = cam_json['fx']
        cam.focal_y = cam_json['fy']
        cam.position = torch.tensor(cam_json['position'])
        cam.rotation = torch.tensor(cam_json['rotation'])
        # End loadCam(

        return cam


    def __iter__(self):
        # 默认一般返回 self 即可
        return self

    def __len__(self):
        return len(self.view_dataset)

    def reset(self):
        self.enumerator = enumerate(self.dataloader)

    def __next__(self):
        try:
            _, (idx, data, loaded_mask, img_shape, (orig_w, orig_h)) = next(self.enumerator)
        except StopIteration:
            self.reset()
            raise StopIteration()
        if self.require_image:
            if loaded_mask.numel() == 0:
                loaded_mask = None
            cam = self.build_camera(idx.item(), data.squeeze(0), loaded_mask)
        else:
            cam = copy.deepcopy(self.cameras_dict[idx.item()])
            cam.image_width = img_shape[2]
            cam.image_height = img_shape[1]
        cam.orig_w = orig_w
        cam.orig_h = orig_h
        return cam


def check_image(cam_info, ppl):
    image = Image.open(cam_info.image_path)
    orig_w, orig_h = image.size
    if ppl.resolution == -1 and orig_w > 1600:
        logger.info("Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.")
        logger.info("If this is not desired, please explicitly specify '--resolution/-r' as 1")


class Scene:

    # gaussians : GaussianModel

    def __init__(self,
                 ppl: PipelineParams,
                 opt: OptimizationParams = None,
                 # gaussians : GaussianModel,
                 # load_iteration=None,
                 # shuffle=True,
                 resolution_scales=(1.0, ),
                 # ply_path=None
                 ):
        """b
        :param path: Path to colmap scene main folder.
        """
        # self.model_path = ppl.model_path
        assert len(resolution_scales) == 1, 'Currently only supports single resolution scale.'
        assert resolution_scales[0] == 1.0, 'Currently only supports resolution_scale==1.'
        self.loaded_iter = None
        # self.gaussians = gaussians

        # if load_iteration:
        #     if load_iteration == -1:
        #         self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
        #     else:
        #         self.loaded_iter = load_iteration
        #
        #     print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.task_train_cameras = {}

        self.x_bound = None
        if os.path.exists(os.path.join(ppl.source_path, "sparse")):
            # Colmap
            logger.info("Found sparse dir, assuming Colmap data set!")
            scene_info = readColmapSceneInfo(ppl.source_path, ppl.images, ppl.data_mode, llffhold=ppl.test_sample_freq, lod=ppl.lod)
        elif os.path.exists(os.path.join(ppl.source_path, "transforms_train.json")):
            # Blender
            logger.info("Found transforms_train.json file, assuming Blender data set!")
            scene_info = readNerfSyntheticInfo(ppl.source_path, ppl.white_background,  ppl.data_mode, ply_path=ppl.ply_data_path)
            self.x_bound = 1.3
        else:
            assert ppl.eval_mode is None, "Cannot evaluate on dataset without SfM"
            logger.info("No SfM info found, try to build SfM using pycolmap")
            source_path = pathlib.Path(ppl.source_path)
            output_folder = source_path.stem + '_colmap'
            output_dir = source_path.parent / output_folder / 'sparse'

            if not output_dir.exists():
                from splatwizard.data_loader.colmap_builder import build_colmap
                output_dir.mkdir(parents=True)
                logger.info(f'Building SfM from {str(source_path)}')
                build_colmap(source_path, output_dir)

            logger.info(f'Using SfM info from {str(output_dir.parent.absolute())}')
            logger.info(f'Using view images from {str(source_path.absolute())}')
            scene_info = readColmapSceneInfo(
                    str(output_dir.parent),
                    str(source_path.absolute()),
                    ppl.data_mode, llffhold=ppl.test_sample_freq, lod=ppl.lod
            )


            # assert False, "Could not recognize scene type!"
        self.scene_info = scene_info
        # if not self.loaded_iter:
        #     if ply_path is not None:
        #         with open(ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
        #             dest_file.write(src_file.read())
        #     else:
        #         with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
        #             dest_file.write(src_file.read())
        #     json_cams = []
        #     camlist = []
        #     if scene_info.test_cameras:
        #         camlist.extend(scene_info.test_cameras)
        #     if scene_info.train_cameras:
        #         camlist.extend(scene_info.train_cameras)
        #     for id, cam in enumerate(camlist):
        #         json_cams.append(camera_to_JSON(id, cam))
        #     with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
        #         json.dump(json_cams, file)

        # if shuffle:
        #     random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # print(f'self.cameras_extent: {self.cameras_extent}')
        check_image(scene_info.train_cameras[0], ppl)
        # for resolution_scale in resolution_scales:
        resolution_scale = resolution_scales[0]

        if  ppl.eval_mode is None:  # Train mode
            logger.info("Building Training Cameras Loader")
            # self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, ppl)
            self.train_cameras[resolution_scale] = CameraIterator(scene_info.train_cameras, ppl, shuffle=True)
            if opt is not None and opt.camera_dependent_task:
                self.task_train_cameras[resolution_scale] = CameraIterator(
                    scene_info.train_cameras, ppl, shuffle=True, require_image=opt.camera_dependent_task_require_images
                )

            logger.info("Building Test Cameras Loader")
            # self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, ppl)
            self.test_cameras[resolution_scale] = CameraIterator(scene_info.test_cameras, ppl)
        else:
            logger.info("Building Test Cameras Loader")
            # self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras,
            #                                                                resolution_scale, ppl)
            self.test_cameras[resolution_scale] = CameraIterator(scene_info.test_cameras,  ppl)
        # elif ppl.data_mode == DataMode.FULL:
        #     logger.info("Loading Train & Test Cameras")
        #     # part1 = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, ppl)
        #     # part2 = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, ppl)
        #     # self.test_cameras[resolution_scale] = part1 + part2
        #
        #     self.test_cameras[resolution_scale] = CameraIterator(scene_info.train_cameras + scene_info.test_cameras, ppl)

        # if self.loaded_iter:
        #     self.gaussians.load_ply_sparse_gaussian(os.path.join(self.model_path,
        #                                                    "point_cloud",
        #                                                    "iteration_" + str(self.loaded_iter),
        #                                                    "point_cloud.ply"))
        #     self.gaussians.load_mlp_checkpoints(os.path.join(self.model_path,
        #                                                    "point_cloud",
        #                                                    "iteration_" + str(self.loaded_iter),
        #                                                    "checkpoint.pth"))
        # else:
        #     self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    # def save(self, iteration):
    #     point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
    #     self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    #     self.gaussians.save_mlp_checkpoints(os.path.join(point_cloud_path, "checkpoint.pth"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def get_task_train_cameras(self, scale=1.0):
        return self.task_train_cameras[scale]