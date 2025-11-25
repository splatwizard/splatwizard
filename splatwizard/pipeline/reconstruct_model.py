import glob
import os
import pathlib
import tempfile

import cv2
# from random import randint
import open3d as o3d
import torch

from splatwizard.modules.dataclass import EvalPack, TrainContext

from splatwizard.model_zoo import init_model
from splatwizard.modules.gaussian_model import GaussianModel
from splatwizard.modules.mesh.extractor import GaussianExtractor, post_process_mesh, cull_scan_mesh, cull_scan_points
from splatwizard.pipeline.evaluation import evaluate
from splatwizard.scene import Scene
from splatwizard.config import PipelineParams, OptimizationParams, ModelParams, EvalMode, MeshExtractorParams
from splatwizard.profiler import profile
from loguru import logger


def reconstruct_model(ppl: PipelineParams, mp: ModelParams, opt: OptimizationParams, mep: MeshExtractorParams, train_context: TrainContext):
    scene = Scene(ppl)
    gs_model: GaussianModel = init_model(train_context.model, mp)



    tb_writer = train_context.tb_writer
    first_iter = 0
    encode_time = 0
    decode_time = 0
    total_bytes = 0
    if ppl.eval_mode == EvalMode.NORMAL:
        assert ppl.checkpoint is not None
        first_iter, _ = gs_model.load(ppl.checkpoint, opt)
        # For some model
        gs_model.final_eval()
        # gs_model.set_state()
    elif ppl.eval_mode == EvalMode.ENCODE_DECODE:
        assert ppl.checkpoint is not None
        first_iter, _ = gs_model.load(ppl.checkpoint, opt)

        with tempfile.NamedTemporaryFile(mode='w+b', delete=True) as tmp_file:
            with profile() as prof:
                gs_model.encode(tmp_file)
            encode_time = prof.duration

            total_bytes = tmp_file.tell()
            # reinitialize model to prevent data leakage
            tmp_file.seek(0)
            gs_model: GaussianModel = init_model(train_context.model, mp)
            gs_model.spatial_lr_scale = scene.cameras_extent
            with profile() as prof:
                gs_model.decode(tmp_file)

            decode_time = prof.duration
    elif ppl.eval_mode == EvalMode.DECODE:
        # reinitialize model to prevent data leakage
        gs_model: GaussianModel = init_model(train_context.model, mp)
        gs_model.spatial_lr_scale = scene.cameras_extent
        total_bytes = os.path.getsize(ppl.bitstream)
        with open(pathlib.Path(ppl.bitstream), 'rb') as f:
            with profile() as prof:
                gs_model.decode(f)
        decode_time = prof.duration


    # with torch.no_grad():
    #     eval_pack: EvalPack = evaluate(gs_model, ppl, scene)
    #     eval_pack.encode_time = encode_time
    #     eval_pack.decode_time = decode_time
    #     eval_pack.total_bytes = total_bytes
    #
    #     gs_model.test_report(eval_pack, first_iter)

    # Using train dataset during reconstruction
    cameras = scene.getTestCameras()

    masks = []
    if ppl.mask_dir:
        # mask_dir = '{0}/mask'.format(instance_dir)
        mask_paths = sorted(glob.glob(os.path.join(ppl.mask_dir, "*.png")))
        for p in mask_paths:
            mask = cv2.imread(p)
            masks.append(mask)

    # 点剔除目前存在一些问题，推测是一些边界外点的问题
    pcd = cull_scan_points(gs_model.xyz.cpu().detach().numpy(), cameras, shrink_factor=mep.shrink_factor, masks=masks)

    if train_context.base_output_dir is not None:
        result_ply_file = train_context.base_output_dir / 'cull_pcd.ply'
        o3d.io.write_point_cloud(result_ply_file, pcd)
        logger.info("culled point cloud saved at {}".format(result_ply_file))

    # exit()
    extractor = GaussianExtractor(gs_model, ppl)
    extractor.gaussians.active_sh_degree = 0



    extractor.reconstruction(cameras)

    assert mep.unbounded is False

    if mep.unbounded:
        name = 'fuse_unbounded.ply'
        mesh = extractor.extract_mesh_unbounded(resolution=mep.mesh_res)
    else:
        name = 'fuse.ply'
        depth_trunc = (extractor.radius * 2.0) if mep.depth_trunc < 0 else mep.depth_trunc
        voxel_size = (depth_trunc / mep.mesh_res) if mep.voxel_size < 0 else mep.voxel_size
        sdf_trunc = 5.0 * voxel_size if mep.sdf_trunc < 0 else mep.sdf_trunc
        mesh = extractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)

    if train_context.base_output_dir is not None:
        mesh_path = train_context.base_output_dir / name
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        logger.info("mesh saved at {}".format(mesh_path))
    # post-process the mesh and save, saving the largest N clusters
    mesh_post = post_process_mesh(mesh, cluster_to_keep=mep.num_cluster)

    if train_context.base_output_dir is not None:
        mesh_post_path = train_context.base_output_dir / name.replace('.ply', '_post.ply')
        o3d.io.write_triangle_mesh(mesh_post_path, mesh_post)
        logger.info("mesh post processed saved at {}".format(mesh_post_path))

        # mesh = o3d.io.read_triangle_mesh(mesh_post_path)

    if mep.unbounded:
        # Culling only supports bounded scene in current stage
        return
    cameras.reset()
    mesh = cull_scan_mesh(mesh_post, cameras, shrink_factor = mep.shrink_factor, masks=masks)

    if train_context.base_output_dir is not None:
        mesh_cull_path = train_context.base_output_dir / name.replace('.ply', '_culled.ply')
        # o3d.io.write_triangle_mesh(mesh_post_path, mesh_post)
        logger.info("culled mesh saved at {}".format(mesh_cull_path))

        o3d.io.write_triangle_mesh(mesh_cull_path, mesh)
