#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#
import glob

import cv2
import torch
import numpy as np
import os
import math

from loguru import logger
from matplotlib import pyplot as plt
from tqdm import tqdm
# from utils.render_utils import save_img_f32, save_img_u8
from functools import partial
import open3d as o3d
import trimesh
import torch.nn.functional as F

from skimage.morphology import binary_dilation, disk
from kornia import morphology as morph

import open3d as o3d
from pathlib import Path
import trimesh


from splatwizard.config import PipelineParams, CullingMode
from splatwizard.modules.mesh import load_K_Rt_from_P
from splatwizard.modules.mesh.marching_cube import marching_cubes_with_contraction
from splatwizard.modules.mesh.utils import open3d_to_trimesh, trimesh_to_open3d
from splatwizard.scene import CameraIterator
from splatwizard.utils.misc import check_tensor
from splatwizard.utils.pose_utils import focus_point_fn


def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    logger.info("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50)  # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    logger.info("num vertices raw {}".format(len(mesh.vertices)))
    logger.info("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        W = viewpoint_cam.image_width
        H = viewpoint_cam.image_height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W - 1) / 2],
            [0, H / 2, 0, (H - 1) / 2],
            [0, 0, 0, 1]]).float().cuda().T
        intrins = (viewpoint_cam.projection_matrix @ ndc2pix)[:3, :3].T
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            cx=intrins[0, 2].item(),
            cy=intrins[1, 2].item(),
            fx=intrins[0, 0].item(),
            fy=intrins[1, 1].item()
        )

        extrinsic = np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj


class GaussianExtractor(object):
    def __init__(self, gaussians, pipe: PipelineParams, bg_color=None):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtractor = GaussianExtractor(gaussians, pipe)
        >>> gaussExtractor.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        if bg_color is None:
            bg_color = [0, 0, 0]
        self.ppl = pipe
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        self.depthmaps = []
        # self.alphamaps = []
        self.rgbmaps = []
        # self.normals = []
        # self.depth_normals = []
        self.viewpoint_stack = []

        self.clean()

    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        # self.alphamaps = []
        self.rgbmaps = []
        # self.normals = []
        # self.depth_normals = []
        self.viewpoint_stack = []

    @torch.no_grad()
    def reconstruction(self, cam_iter):
        """
        reconstruct radiance field given cameras
        """
        self.clean()
        self.viewpoint_stack = cam_iter
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            render_pkg = self.gaussians.render(viewpoint_cam, self.background, self.ppl)
            rgb = render_pkg.rendered_image
            # alpha = render_pkg['rend_alpha']
            # normal = torch.nn.functional.normalize(render_pkg['rend_normal'], dim=0)
            depth = render_pkg.surf_depth
            # depth[(cam.gt_alpha_mask < 0.5)] = 0
            # check_tensor(depth)

            # depth = (1/depth).clamp(0, 10)
            # plt.imshow(depth.squeeze(0).detach().cpu().numpy())
            # plt.colorbar()
            # plt.show()
            # exit()
            # depth_normal = render_pkg['surf_normal']
            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())
            # self.alphamaps.append(alpha.cpu())
            # self.normals.append(normal.cpu())
            # self.depth_normals.append(depth_normal.cpu())

        # self.rgbmaps = torch.stack(self.rgbmaps, dim=0)
        # self.depthmaps = torch.stack(self.depthmaps, dim=0)
        # self.alphamaps = torch.stack(self.alphamaps, dim=0)
        # self.depth_normals = torch.stack(self.depth_normals, dim=0)
        self.estimate_bounding_sphere()

    def estimate_bounding_sphere(self):
        """
        Estimate the bounding sphere given camera pose
        """
        # from utils.render_utils import transform_poses_pca, focus_point_fn
        torch.cuda.empty_cache()
        c2ws = np.array(
            [np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in self.viewpoint_stack])
        poses = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1])
        center = (focus_point_fn(poses))
        self.radius = np.linalg.norm(c2ws[:, :3, 3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()
        logger.info(f"The estimated bounding radius is {self.radius:.2f}")
        logger.info(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")

    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True, masks=None):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.

        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        logger.info("Running tsdf volume integration ...")
        logger.info(f'voxel_size: {voxel_size}')
        logger.info(f'sdf_trunc: {sdf_trunc}')
        logger.info(f'depth_truc: {depth_trunc}')

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for i, (cam_o3d, cam) in tqdm(enumerate(zip(to_cam_open3d(self.viewpoint_stack), self.viewpoint_stack)), desc="TSDF integration progress"):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]

            # if we have mask provided, use it
            if mask_backgrond and (cam.gt_alpha_mask is not None):
                depth[(cam.gt_alpha_mask < 0.5)] = 0

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(
                    np.asarray(np.clip(rgb.permute(1, 2, 0).cpu().numpy(), 0.0, 1.0) * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1, 2, 0).cpu().numpy(), order="C")),
                depth_trunc=depth_trunc, convert_rgb_to_intensity=False,
                depth_scale=1.0
            )

            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
        return mesh

    @torch.no_grad()
    def extract_mesh_unbounded(self, resolution=1024):
        """
        Experimental features, extracting meshes from unbounded scenes, not fully test across datasets.
        return o3d.mesh
        """

        def contract(x):
            mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))

        def uncontract(y):
            mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, y, (1 / (2 - mag) * (y / mag)))

        def compute_sdf_perframe(i, points, depthmap, rgbmap, viewpoint_cam):
            """
                compute per frame sdf
            """
            new_points = torch.cat([points, torch.ones_like(points[..., :1])],
                                   dim=-1) @ viewpoint_cam.full_proj_transform
            z = new_points[..., -1:]
            pix_coords = (new_points[..., :2] / new_points[..., -1:])
            mask_proj = ((pix_coords > -1.) & (pix_coords < 1.) & (z > 0)).all(dim=-1)
            sampled_depth = torch.nn.functional.grid_sample(depthmap.cuda()[None], pix_coords[None, None],
                                                            mode='bilinear', padding_mode='border',
                                                            align_corners=True).reshape(-1, 1)
            sampled_rgb = torch.nn.functional.grid_sample(rgbmap.cuda()[None], pix_coords[None, None], mode='bilinear',
                                                          padding_mode='border', align_corners=True).reshape(3, -1).T
            sdf = (sampled_depth - z)
            return sdf, sampled_rgb, mask_proj

        def compute_unbounded_tsdf(samples, inv_contraction, voxel_size, return_rgb=False):
            """
                Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
            """
            if inv_contraction is not None:
                mask = torch.linalg.norm(samples, dim=-1) > 1
                # adaptive sdf_truncation
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1 / (2 - torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
                samples = inv_contraction(samples)
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:, 0]) * (-1)
            rgbs = torch.zeros((samples.shape[0], 3)).cuda()

            weights = torch.ones_like(samples[:, 0])
            for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="TSDF integration progress"):
                sdf, rgb, mask_proj = compute_sdf_perframe(i, samples,
                                                           depthmap=self.depthmaps[i],
                                                           rgbmap=self.rgbmaps[i],
                                                           viewpoint_cam=self.viewpoint_stack[i],
                                                           )

                # volume integration
                sdf = sdf.flatten()
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]
                wp = w + 1
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:, None] + rgb[mask_proj]) / wp[:, None]
                # update weight
                weights[mask_proj] = wp

            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        normalize = lambda x: (x - self.center) / self.radius
        unnormalize = lambda x: (x * self.radius) + self.center
        inv_contraction = lambda x: unnormalize(uncontract(x))

        N = resolution
        voxel_size = (self.radius * 2 / N)
        logger.info(f"Computing sdf gird resolution {N} x {N} x {N}")
        logger.info(f"Define the voxel_size as {voxel_size}")
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)
        # from utils.mcube_utils import marching_cubes_with_contraction
        R = contract(normalize(self.gaussians.get_xyz)).norm(dim=-1).cpu().numpy()
        R = np.quantile(R, q=0.95)
        R = min(R + 0.01, 1.9)

        mesh = marching_cubes_with_contraction(
            sdf=sdf_function,
            bounding_box_min=(-R, -R, -R),
            bounding_box_max=(R, R, R),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
        )

        # coloring the mesh
        torch.cuda.empty_cache()
        mesh = mesh.as_open3d
        logger.info("texturing mesh ... ")
        _, rgbs = compute_unbounded_tsdf(torch.tensor(np.asarray(mesh.vertices)).float().cuda(), inv_contraction=None,
                                         voxel_size=voxel_size, return_rgb=True)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
        return mesh

    # @torch.no_grad()
    # def export_image(self, path):
    #     render_path = os.path.join(path, "renders")
    #     gts_path = os.path.join(path, "gt")
    #     vis_path = os.path.join(path, "vis")
    #     os.makedirs(render_path, exist_ok=True)
    #     os.makedirs(vis_path, exist_ok=True)
    #     os.makedirs(gts_path, exist_ok=True)
    #     for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
    #         gt = viewpoint_cam.original_image[0:3, :, :]
    #         save_img_u8(gt.permute(1, 2, 0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    #         save_img_u8(self.rgbmaps[idx].permute(1, 2, 0).cpu().numpy(),
    #                     os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    #         save_img_f32(self.depthmaps[idx][0].cpu().numpy(),
    #                      os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".tiff"))
    #         # save_img_u8(self.normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'normal_{0:05d}'.format(idx) + ".png"))
    #         # save_img_u8(self.depth_normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'depth_normal_{0:05d}'.format(idx) + ".png"))


def cull_scan_mesh(mesh, cam_iter, shrink_factor=1., masks=None, skip_culiing=False, culling_mode=CullingMode.VIEW_KEEP):

        # load poses
        # image_dir = '{0}/images'.format(instance_dir)
        # image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        # n_images = len(image_paths)
        # cam_file = '{0}/cameras.npz'.format(instance_dir)
        # camera_dict = np.load(cam_file)
        # scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
        # world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

        mesh = open3d_to_trimesh(mesh)

        assert cam_iter.shuffle is False
        intrinsics_all = []
        pose_all = []
        for cam in cam_iter:
            P = cam.world_mat @ cam.scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            intrinsics_all.append(torch.from_numpy(intrinsics).float())
            pose_all.append(torch.from_numpy(pose).float())

        # load transformation matrix

        vertices = mesh.vertices

        # project and filter
        vertices = torch.from_numpy(vertices).cuda()
        vertices = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=-1)
        vertices = vertices.permute(1, 0)
        vertices = vertices.float()

        if skip_culiing:
            camera = next(cam_iter)
            scale_mat = camera.scale_mat
            mesh.vertices = mesh.vertices * scale_mat[0, 0] + scale_mat[:3, 3][None]
            cam_iter.reset()

            return trimesh_to_open3d(mesh)

        sampled_masks = []
        for i, cam in tqdm(enumerate(cam_iter), desc="Culling mesh given masks"):
            pose = pose_all[i]
            w2c = torch.inverse(pose).cuda()
            intrinsic = intrinsics_all[i].cuda()

            # if masks:
            #     maski = masks[i][:, :, 0].astype(np.float32) / 256.
            # else:
            #     maski = cams[i].gt_alpha_mask.squeeze(0).cpu().numpy()
            # mask = cams[i].gt_alpha_mask

            if masks:
                maski = masks[i][:, :, 0].astype(np.float32) / 256.
                maski = torch.Tensor(maski).cuda()
            else:
                maski = cam.gt_alpha_mask.squeeze(0)  # .cpu().numpy()

            W = cam.orig_w.cuda() * shrink_factor
            H = cam.orig_h.cuda() * shrink_factor

            with torch.no_grad():
                # transform and project
                cam_points = intrinsic @ w2c @ vertices
                pix_coords = cam_points[:2, :] / (cam_points[2, :].unsqueeze(0) + 1e-6)
                pix_coords = pix_coords.permute(1, 0)

                pix_coords[..., 0] /= W - 1
                pix_coords[..., 1] /= H - 1
                pix_coords = (pix_coords - 0.5) * 2
                valid = ((pix_coords > -1.) & (pix_coords < 1.)).all(dim=-1).float()

                # dialate mask similar to unisurf
                # maski = mask[i][:, :, 0].astype(np.float32) / 256.
                # maski = mask.squeeze(0).cpu().numpy()
                # maski = torch.from_numpy(binary_dilation(maski, disk(24))).float()[None, None].cuda()
                maski = morph.dilation(maski.unsqueeze(0).unsqueeze(0).cuda(), torch.Tensor(disk(24)).cuda())

                sampled_mask = \
                    F.grid_sample(maski, pix_coords[None, None], mode='nearest', padding_mode='zeros',
                                  align_corners=True)[
                        0, -1, 0]
                if culling_mode == CullingMode.VIEW_KEEP:
                    sampled_mask = sampled_mask * valid
                else:
                    sampled_mask = sampled_mask + 1. - valid
                sampled_masks.append(sampled_mask)

        sampled_masks = torch.stack(sampled_masks, -1)
        # filter
        if culling_mode == CullingMode.VIEW_KEEP:
            mask = (sampled_masks > 0.).any(dim=-1).cpu().numpy()
        else:
            mask = (sampled_masks > 0.).all(dim=-1).cpu().numpy()

        # print(mask.mean())
        face_mask = mask[mesh.faces].all(axis=1)

        mesh.update_vertices(mask)
        mesh.update_faces(face_mask)

        # transform vertices to world
        # scale_mat = scale_mats[0]
        camera = next(cam_iter)
        scale_mat = camera.scale_mat
        mesh.vertices = mesh.vertices * scale_mat[0, 0] + scale_mat[:3, 3][None]
        cam_iter.reset()

        return trimesh_to_open3d(mesh)


def cull_scan_points(pcd, cam_iter: CameraIterator, shrink_factor=1., masks=None, skip_culling=False, culling_mode: CullingMode = CullingMode.VIEW_KEEP):

    # load poses
    # image_dir = '{0}/images'.format(instance_dir)
    # image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    # n_images = len(image_paths)
    # cam_file = '{0}/cameras.npz'.format(instance_dir)
    # camera_dict = np.load(cam_file)
    # scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    # world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    # cams = list(cam_iter)
    assert cam_iter.shuffle is False
    intrinsics_all = []
    pose_all = []
    for cam in cam_iter:
        P = cam.world_mat @ cam.scale_mat
        P = P[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(None, P)
        intrinsics_all.append(torch.from_numpy(intrinsics).float())
        pose_all.append(torch.from_numpy(pose).float())

    # load mask
    # mask_dir = '{0}/mask'.format(instance_dir)
    # mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    # masks = []
    # for p in mask_paths:
    #     mask = cv2.imread(p)
    #     masks.append(mask)

    # hard-coded image shape
    # W, H = 1600, 1200

    # load pcd
    # pcd = o3d.io.read_point_cloud(ply_path)
    # load transformation matrix

    vertices = pcd # np.asarray(pcd.points)

    # project and filter
    vertices = torch.from_numpy(vertices).cuda()
    vertices = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=-1)
    vertices = vertices.permute(1, 0)
    vertices = vertices.float()
    if skip_culling:
        camera = next(cam_iter)
        points = pcd  # np.asarray(pcd.points)
        scale_mat = camera.scale_mat
        points = points * scale_mat[0, 0] + scale_mat[:3, 3][None]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        cam_iter.reset()
        return pcd

    sampled_masks = []
    for i, cam in tqdm(enumerate(cam_iter),  desc="Culling point clouds given masks"):
        # if i == 21:
        #     continue

        pose = pose_all[i]
        w2c = torch.inverse(pose).cuda()
        intrinsic = intrinsics_all[i].cuda()

        if masks:
            maski = masks[i][:, :, 0].astype(np.float32) / 256.
            maski = torch.Tensor(maski).cuda()
        else:
            maski = cam.gt_alpha_mask.squeeze(0)  # .cpu().numpy()

        W = cam.orig_w.cuda() * shrink_factor
        H = cam.orig_h.cuda() * shrink_factor


        with torch.no_grad():
            # transform and project
            cam_points = intrinsic @ w2c @ vertices
            pix_coords = cam_points[:2, :] / (cam_points[2, :].unsqueeze(0) + 1e-6)
            pix_coords = pix_coords.permute(1, 0)
            pix_coords[..., 0] /= W - 1
            pix_coords[..., 1] /= H - 1
            pix_coords = (pix_coords - 0.5) * 2
            valid = ((pix_coords > -1. ) & (pix_coords < 1.)).all(dim=-1).float()

            # print(i)
            # plt.scatter(pix_coords[valid.bool(), 0].detach().cpu().numpy(), pix_coords[valid.bool() , 1].detach().cpu().numpy(), s=0.001)
            # plt.show()
            # exit()

            # dialate mask similar to unisurf
            # maski = masks[i][:, :, 0].astype(np.float32) / 256.
            # maski = mask.squeeze(0).cpu().numpy()
            # maski = torch.from_numpy(binary_dilation(maski, disk(24))).float()[None, None].cuda()
            maski = morph.dilation(maski.unsqueeze(0).unsqueeze(0).cuda(), torch.Tensor(disk(24)).cuda())

            sampled_mask = F.grid_sample(maski, pix_coords[None, None], mode='nearest', padding_mode='zeros', align_corners=True)[0, -1, 0]

            # plt.scatter(pix_coords[(valid * sampled_mask).bool(), 0].detach().cpu().numpy(), pix_coords[(valid * sampled_mask).bool() , 1].detach().cpu().numpy(), s=0.001)
            # plt.show()

            if culling_mode == CullingMode.VIEW_KEEP:
                sampled_mask = sampled_mask * valid
            else:
                sampled_mask = sampled_mask + (1. - valid)
            sampled_masks.append(sampled_mask)

            # valid = sampled_mask > 0


            # exit()





    sampled_masks = torch.stack(sampled_masks, -1)
    # filter
    # check_tensor(sampled_masks)
    if culling_mode == CullingMode.VIEW_KEEP:
        mask = (sampled_masks > 0.).any(dim=-1).cpu().numpy()
    else:
        mask = (sampled_masks > 0.).all(dim=-1).cpu().numpy()

    # print(mask.mean())
    points = pcd # np.asarray(pcd.points)
    points = points[mask]
    # transform vertices to world
    # scale_mat = scale_mats[0]

    camera = next(cam_iter)
    scale_mat = camera.scale_mat
    points = points * scale_mat[0, 0] + scale_mat[:3, 3][None]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    cam_iter.reset()
    return pcd
    # o3d.io.write_point_cloud(result_ply_file, pcd)