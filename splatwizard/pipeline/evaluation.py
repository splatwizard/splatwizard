import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
import torchvision

from splatwizard.config import PipelineParams
from splatwizard.data_loader.dataset_readers import fetchPly
from splatwizard.metrics.loss_utils import l1_func, ssim_func, psnr_func, lpips_fn
from splatwizard.modules.dataclass import EvalPack, TrainContext
from splatwizard.modules.gaussian_model import  GaussianModel

from splatwizard.scene import Scene
from splatwizard.profiler import profile



@torch.no_grad()
def evaluate(
        gs_model: GaussianModel,
        ppl: PipelineParams,
        scene: Scene,
        ppl_context: TrainContext=None):

    # if tb_writer:
    #     tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
    #     tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
    #     tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)
    #
    # if wandb is not None:
    #     wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    # # Report test and samples of training set
    # if iteration in testing_iterations:
    # print('start evaluation')
    gs_model.eval()

    # validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
    #                               {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

    bg_color = [1, 1, 1] if ppl.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=ppl.device)

    l1_test = 0.0
    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0
    total_time = 0.0
    accumulate_gaussians = 0
    total_gaussians = 0
    peak_memory_allocated = 0
    peak_memory_reserved = 0

    # if ppl.ref_ply_path is not None:
    #     ref_ply = fetchPly(ppl.ref_ply_path)
    #     model_points = np.array(gs_model.xyz.detach().cpu().numpy(), order='C')
    #     check_tensor(torch.Tensor(model_points))
    #     ref_points = np.array(ref_ply.points, order='C')
    #     check_tensor(torch.Tensor(ref_points))
    #     chamfer_dist = pcu.chamfer_distance(model_points, ref_points, p_norm=1)
    #     print('chamfer distance:', chamfer_dist)


    cameras = scene.getTestCameras()

    if ppl.eval_warmup and ppl.eval_mode is not None:
        logger.info('Warmup model')
        viewpoint = next(cameras)
        render_result = gs_model.render(viewpoint, background, ppl)

        # for idx, viewpoint in enumerate(cameras):
        #     render_result = gs_model.render(viewpoint, background, ppl)
        # print("[DEBUG] Rendered image mean:", render_result.rendered_image.mean().item())
        # plt.imshow(render_result.rendered_image.permute(1, 2, 0).cpu().detach().numpy())
        # plt.show()
        # exit()
        cameras.reset()

    try:
        total_gaussians = gs_model.xyz.shape[0]
    except:
        pass

    # logger.info('Start evaluation')
    for idx, viewpoint in enumerate(cameras):
        with profile() as prof:
            render_result = gs_model.render(viewpoint, background, ppl)

        gt_image = torch.clamp(viewpoint.original_image.to(render_result.rendered_image.device), 0.0, 1.0)

        # plt.imshow(gt_image.permute(1, 2, 0).cpu().detach().numpy())
        # plt.show()
        # plt.imshow(render_result.rendered_image.permute(1, 2, 0).cpu().detach().numpy())
        # plt.show()
        # exit()

        if ppl_context is not None and ppl.save_rendered_image:
            torchvision.utils.save_image(render_result.rendered_image, ppl_context.render_result_dir / ('{0:05d}'.format(idx) + ".png"))

        l1_loss = l1_func(render_result.rendered_image, gt_image)
        ssim_loss = ssim_func(render_result.rendered_image, gt_image)
        psnr_val = psnr_func(render_result.rendered_image, gt_image)
        lpips_val = lpips_fn(render_result.rendered_image, gt_image, normalize=True)

        l1_test += l1_loss.mean().double().item()
        psnr_test += psnr_val.mean().double().item()
        ssim_test += ssim_loss.mean().double().item()
        lpips_test += lpips_val.mean().double().item()
        # lpips_test += 0 #lpips(image, gt_image, net_type='vgg').detach().mean().double()
        total_time += prof.duration
        accumulate_gaussians += render_result.visibility_filter.sum().item() if render_result.visibility_filter is not None else 0

        peak_memory_allocated = max(peak_memory_allocated, prof.peak_memory_allocated)
        peak_memory_reserved = max(peak_memory_reserved, prof.peak_memory_reserved)

        # torch.cuda.empty_cache()




    total_cameras = len(cameras)
    psnr_test /= total_cameras
    ssim_test /= total_cameras
    lpips_test /= total_cameras
    l1_test /= total_cameras
    frame_time = total_time / total_cameras
    avg_gaussians = accumulate_gaussians / total_cameras

    eval_result = gs_model.eval_pack_cls(
        l1_val=l1_test,
        psnr_val=psnr_test,
        ssim_val=ssim_test,
        lpips_val=lpips_test,
        frame_time=frame_time,
        peak_memory_allocated=peak_memory_allocated,
        peak_memory_reserved=peak_memory_reserved,
        avg_gaussians=avg_gaussians,
        total_gaussian=total_gaussians
    )
    # print(  'Evaluating test',
    #         'L1:', l1_test,
    #         'PSNR', psnr_test,
    #         'SSIM', ssim_test,
    #         'LPIPS', lpips_test,
    #         'FPS', 1 / frame_time)

    eval_result = gs_model.post_eval(eval_result)
    gs_model.train()
    return eval_result
