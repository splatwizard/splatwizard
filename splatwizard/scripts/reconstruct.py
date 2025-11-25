
import pathlib
import sys
from enum import Enum

from simple_parsing import ArgumentParser
from loguru import logger


from splatwizard.modules.dataclass import TrainContext

from splatwizard.config import PipelineParams, OptimizationParams, EvalMode, MeshExtractorParams
from splatwizard.model_zoo import CONFIG_CACHE
from splatwizard.pipeline.reconstruct_model import reconstruct_model

from splatwizard.utils.logging import setup_tensorboard
from splatwizard.utils.misc import safe_state


def setup_output_dir(pp: PipelineParams, train_context: TrainContext, stage: Enum=None):
    # 创建输出目录

    if pp.output_dir is None:
        return
    if stage is not None:
        train_context.output_dir = train_context.base_output_dir / str(stage.name)
        train_context.output_dir.mkdir(exist_ok=True)
    else:
        train_context.output_dir = train_context.base_output_dir

    train_context.checkpoint_dir = train_context.output_dir / 'checkpoints'
    train_context.checkpoint_dir.mkdir(exist_ok=True)

    # if pp.env_path is not None:
    #     os.environ["PATH"] = pp.env_path + ':' + os.environ["PATH"]

    # 初始化日志文件
    logger.add(train_context.output_dir / 'recon.log')
    # logger.info(f'running tag: {args.tag}')

    if pp.dataset is None:
        dataset = pathlib.Path(pp.source_path).name
    else:
        dataset = pp.dataset

    tb_writer = setup_tensorboard(train_context.output_dir)
    tb_writer.prefix = dataset

    train_context.tb_writer = tb_writer

    logger.info("Setting up output dir" + str(train_context.output_dir.absolute()))

    return train_context


def validate_pipeline_parameters(pp: PipelineParams):
    if pp.lanczos_resample:
        assert pp.num_workers == 0, "Only single worker mode supports lanczos_resample=True"

    assert pp.checkpoint_type in ('pth', 'ply'), f'Unsupported checkpoint type: {pp.checkpoint_type}'

    if isinstance(pp.final_checkpoint, str):
        assert pp.final_checkpoint in ('pth', 'ply'), f'Unsupported checkpoint type: {pp.checkpoint_type}'
    else:
        for type_ in pp.final_checkpoint:
            assert type_ in ('pth', 'ply'), f'Unsupported checkpoint type: {pp.checkpoint_type}'


def main():
    parser = ArgumentParser(add_config_path_arg=True)
    parser.add_arguments(PipelineParams, dest="pipeline")  # noqa
    parser.add_arguments(CONFIG_CACHE[0], dest="model_group")  # noqa
    parser.add_arguments(MeshExtractorParams, dest="mesh_group") # noqa

    args = parser.parse_args(sys.argv[1:])

    mp = args.model_group.model
    pp: PipelineParams = args.pipeline
    op: OptimizationParams = args.model_group.optim
    mep: MeshExtractorParams = args.mesh_group

    validate_pipeline_parameters(pp)

    if pp.seed is not None:
        safe_state(pp.seed)
    train_context = TrainContext()
    train_context.model = args.subgroups['model_group.model']

    if pp.output_dir is not None:
        # 创建输出目录
        train_context.base_output_dir = pathlib.Path(pp.output_dir)
        train_context.base_output_dir.mkdir(exist_ok=True, parents=True)

        logger.info("Output dir: " + str(train_context.base_output_dir.absolute()))

    if pp.eval_mode is None:
        pp.eval_mode = EvalMode.NORMAL



    setup_output_dir(pp, train_context)
    logger.info(f"{pp}")
    logger.info(f"{mp}")
    logger.info(f"{op}")
    reconstruct_model(pp, mp, op, mep,  train_context)

if __name__ == '__main__':
    sys.exit(main())
