import copy
from random import randint

from loguru import logger
import torch

from splatwizard.model_zoo import init_model
from splatwizard.modules.dataclass import TrainContext
from splatwizard.modules.gaussian_model import GaussianModel
from splatwizard.pipeline.evaluation import evaluate
from splatwizard.scene import Scene
from splatwizard.config import PipelineParams, OptimizationParams, ModelParams
from splatwizard.scheduler import Scheduler
from splatwizard.profiler import profile


def train_model(ppl: PipelineParams, mp: ModelParams, opt: OptimizationParams, train_context: TrainContext, prev_model=None):
    scene = Scene(ppl, opt)
    if prev_model is None:
        gs_model: GaussianModel = init_model(train_context.model, mp)
        if mp.require_cam_infos:
            gs_model.create_from_pcd(scene.scene_info.point_cloud, scene.cameras_extent, scene.train_cameras[1.0])
        else:
            gs_model.create_from_pcd(scene.scene_info.point_cloud, scene.cameras_extent)
    else:
        logger.info(f'Using input model')
        gs_model = prev_model
    # Manage tasks executed before rendering
    pre_scheduler = Scheduler()
    gs_model.register_pre_task(pre_scheduler, ppl, opt)

    # Manage tasks executed after rendering
    post_scheduler = Scheduler()
    gs_model.register_post_task(post_scheduler, ppl, opt)

    first_iter = 0

    if ppl.checkpoint:
        logger.info(f'Using checkpoint {str(ppl.checkpoint)}')
        first_iter, need_setup = gs_model.load(ppl.checkpoint, opt=opt)
        assert first_iter is not None, "Cannot use .ply for checkpoint in training, Using --init_checkpoint for multi-stage training"
    elif ppl.init_checkpoint:
        logger.info(f'Using init_checkpoint {str(ppl.init_checkpoint)}')
        _, need_setup = gs_model.load(ppl.init_checkpoint, opt=opt)
    else:
        need_setup = True

    if need_setup or opt.force_setup:
        gs_model.training_setup(opt)

    gs_model.after_setup_hook(ppl, opt)

    first_iter += 1
    pre_scheduler.init(first_iter)
    post_scheduler.init(first_iter)

    bg_color = [1, 1, 1] if ppl.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=ppl.device)

    gs_model.train()
    cam_iterator = scene.getTrainCameras()

    if opt.camera_dependent_task:
        task_cam_iterator = scene.get_task_train_cameras()
    else:
        task_cam_iterator = None

    for iteration in range(first_iter, opt.iterations + 1):
        with profile(skip=not ppl.profile_train) as pre_prof:
            pre_scheduler.exec_task(ppl, opt, cam_iterator=task_cam_iterator)
            pre_scheduler.step()

        # Pick a random Camera
        # if not viewpoint_stack:
        #     viewpoint_stack = scene.getTrainCameras().copy()
        # viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        try:
            viewpoint_cam = next(cam_iterator)
        except StopIteration:
            # cam_iterator.reset()
            viewpoint_cam = next(cam_iterator)

        with profile(skip=not ppl.profile_train) as prof1:
            bg = torch.rand((3,), device=ppl.device) if opt.random_background else background
            render_result = gs_model.render(viewpoint_cam, bg, ppl, opt, step=iteration)
            loss, loss_pack = gs_model.loss_func(viewpoint_cam, render_result, opt)
            loss.backward()

        with torch.no_grad():
            if ppl.eval_freq is not None and (iteration + 1) % ppl.eval_freq == 0:
                eval_pack = evaluate(gs_model, ppl, scene)
                gs_model.eval_report(eval_pack, iteration, train_context.tb_writer)

            with profile(skip=not ppl.profile_train) as post_prof:
                post_scheduler.exec_task(ppl, opt, render_result, task_cam_iterator)
                post_scheduler.step()

            with profile(skip=not ppl.profile_train) as prof2:
                if iteration < opt.iterations:
                    gs_model.optimizer_step(render_result, opt, step=iteration)
                    # gs_model.optimizer.zero_grad(set_to_none=True)

            loss_pack.train_elapsed_time = prof1.duration + prof2.duration
            loss_pack.task_elapsed_time = pre_prof.duration + post_prof.duration
            loss_pack.peak_memory_allocated = max(
                prof1.peak_memory_allocated,
                prof2.peak_memory_allocated,
                pre_prof.peak_memory_allocated,
                post_prof.peak_memory_allocated
            )

            loss_pack.peak_memory_reserved = max(
                prof1.peak_memory_reserved,
                prof2.peak_memory_reserved,
                pre_prof.peak_memory_reserved,
                post_prof.peak_memory_reserved
            )

            gs_model.train_report(loss_pack, iteration, train_context.tb_writer)

            if train_context.checkpoint_dir is not None:
                if iteration in ppl.checkpoint_iterations:
                    gs_model.save(train_context.checkpoint_dir, iteration, type_=ppl.checkpoint_type)
                elif iteration == opt.iterations:   # save last checkpoint
                    if isinstance(ppl.final_checkpoint, str):
                        gs_model.save(train_context.checkpoint_dir, iteration, type_=ppl.checkpoint_type)
                    else:
                        for type_ in ppl.final_checkpoint:
                            gs_model.save(train_context.checkpoint_dir, iteration, type_=type_)



    return gs_model