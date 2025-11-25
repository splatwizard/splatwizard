import json
import os
import pathlib
import tempfile
# from random import randint

import torch
from splatwizard.modules.dataclass import EvalPack

from splatwizard.model_zoo import init_model
from splatwizard.modules.gaussian_model import GaussianModel
from splatwizard.pipeline.evaluation import evaluate
from splatwizard.scene import Scene
from splatwizard.config import PipelineParams, OptimizationParams, ModelParams, EvalMode
from splatwizard.profiler import profile
from loguru import logger


def eval_model(ppl: PipelineParams, mp: ModelParams, opt: OptimizationParams, train_context):
    scene = Scene(ppl)
    gs_model: GaussianModel = init_model(train_context.model, mp)
    # gs_model.create_from_pcd(scene.scene_info.point_cloud, scene.cameras_extent)

    tb_writer = train_context.tb_writer
    first_iter = 0
    encode_time = 0
    decode_time = 0
    total_bytes = 0
    if ppl.eval_mode == EvalMode.NORMAL:
        assert ppl.checkpoint is not None
        first_iter, _ = gs_model.load(ppl.checkpoint, opt)
        gs_model.final_eval()
        # gs_model.set_state()
    elif ppl.eval_mode == EvalMode.ENCODE_DECODE:
        assert ppl.checkpoint is not None
        first_iter, _ = gs_model.load(ppl.checkpoint, opt)

        if ppl.save_bitstream:
            with open(train_context.base_output_dir / 'encoded.bin', 'wb') as f:
                with profile() as prof:
                    gs_model.encode(f)
                encode_time = prof.duration
                total_bytes = f.tell()


            gs_model: GaussianModel = init_model(train_context.model, mp)
            gs_model.spatial_lr_scale = scene.cameras_extent
            with open(train_context.base_output_dir / 'encoded.bin', 'rb') as f:
                with profile() as prof:
                    gs_model.decode(f)

                decode_time = prof.duration

        else:
            with tempfile.NamedTemporaryFile(mode='w+b', delete=True) as tmp_file:
                with profile() as prof:
                    gs_model.encode(tmp_file)
                encode_time = prof.duration
                # exit()
                total_bytes = tmp_file.tell()
                # # reinitialize model to prevent data leakage
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


    with torch.no_grad():
        eval_pack: EvalPack = evaluate(gs_model, ppl, scene, train_context)
        eval_pack.encode_time = encode_time
        eval_pack.decode_time = decode_time
        eval_pack.total_bytes = total_bytes

        results = gs_model.test_report(eval_pack, first_iter)
        if results is not None:
            with open(train_context.base_output_dir / 'results.json', 'w') as f:
                json.dump(results, f)



