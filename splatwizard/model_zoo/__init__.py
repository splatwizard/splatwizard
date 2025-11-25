from dataclasses import dataclass

from simple_parsing import subgroups

from ..config import ModelParams, OptimizationParams
from .registry import MODEL_REGISTRY
from loguru import logger

def build_opt_group(model_registry):
    opt_groups = {}
    for k, v in model_registry.items():
        if isinstance(v[1], dict):
            for sub_k, sub_v in v[1].items():
                # print('sub_k', sub_k, 'sub_v', sub_v)
                opt_groups[k + '_' + sub_k] = sub_v
        else:
            opt_groups[k] = v[1]

    return opt_groups


@dataclass
class GroupConfig:
    model: ModelParams = subgroups(
        {k: v[0] for k, v in MODEL_REGISTRY.items() },
        default="3dgs",
    )

    optim: OptimizationParams = subgroups(
        build_opt_group(MODEL_REGISTRY),
        default="3dgs",
    )


def init_model(model_name, mp: ModelParams):
    gs_model = MODEL_REGISTRY[model_name][2](mp)
    return gs_model


def register_model(name, model_param_cls, opt_param_cls, model_cls):
    assert name not in MODEL_REGISTRY.keys()
    MODEL_REGISTRY[name] = (model_param_cls, opt_param_cls, model_cls)

    @dataclass
    class GroupConfig:
        model: ModelParams = subgroups(
            {k: v[0] for k, v in MODEL_REGISTRY.items()},
            default="3dgs",
        )

        optim: OptimizationParams = subgroups(
            build_opt_group(MODEL_REGISTRY),
            default="3dgs",
        )
    CONFIG_CACHE[0] = GroupConfig


CONFIG_CACHE = {0: GroupConfig}
