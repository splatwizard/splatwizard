import importlib

from .cat_3dgs.config import CAT3DGSModelParams, CAT3DGSOptimizationParams
from .gs.config import GSModelParams, GSOptimizationParams
from .gs_dr_aa.config import GSDRAAOptimizationParams, GSDRAAModelParams
from .hac.config import HACModelParams, HACOptimizationParams
from .lightgaussian.config import LightGaussianParams, LightGaussianPruneOptimizationParams, \
    LightGaussianDistillOptimizationParams, LightGaussianEncodeOptimizationParams
from .chimerags.config import ChimeraGSDistillOptimizationParams, ChimeraGSEncodeOptimizationParams, ChimeraGSModelParams
from .pup3dgs.config import PUP3DGSModelParams, PUP3DGSOptimizationParams
from .speedy_splat.config import SpeedySplatModelParams, SpeedySplatOptimizationParams
from .trim3dgs.config import Trim3DGSModelParams, Trim3DGSOptimizationParams
from .trimming_the_fat.config import TTFModelParams, TTFOptimizationParams
from .compact3dgs.config import CompactGSModelParams, CompactGSOptimizationParams
from .c3dgs.config import C3DGSModelParams, C3DGSOptimizationParams
from .controlgs.config import ControlGSModelParams, ControlGSOptimizationParams
from .surfel_gs.config import SurfelGSModelParams, SurfelGSOptimizationParams
from .mesongs.config import MesonGSModelParams, MesonGSOptimizationParams


def dynamic_load(import_path, cls_name, prefix='splatwizard.model_zoo.'):
    def wrapper(*args, **kwargs):
        if prefix is not None:
            path = prefix + import_path
        else:
            path = import_path
        # Dynamically load a module
        module = importlib.import_module(path)

        # Access a class within the loaded module
        cls = getattr(module, cls_name)
        return cls(*args, **kwargs)
    return wrapper


MODEL_REGISTRY = {
    "3dgs": (GSModelParams, GSOptimizationParams, dynamic_load('gs.model', 'GSModel')),
    "hac": (HACModelParams, HACOptimizationParams, dynamic_load('hac.model', 'HAC')),
    "cat3dgs": (CAT3DGSModelParams, CAT3DGSOptimizationParams, dynamic_load('cat_3dgs.model', 'CAT3DGS')),
    "trimming_the_fat": (TTFModelParams, TTFOptimizationParams, dynamic_load('trimming_the_fat.model', 'TTF')),
    "compactgs": (CompactGSModelParams, CompactGSOptimizationParams, dynamic_load('compact3dgs.model', 'CompactGSModel')),
    "c3dgs": (C3DGSModelParams, C3DGSOptimizationParams, dynamic_load('c3dgs.model', 'C3DGS')),
    "controlgs": (ControlGSModelParams, ControlGSOptimizationParams, dynamic_load('controlgs.model', 'ControlGS')),
    "lightgaussian": (LightGaussianParams,
                      {'prune': LightGaussianPruneOptimizationParams,
                        'distill': LightGaussianDistillOptimizationParams,
                       'encode': LightGaussianEncodeOptimizationParams
                       },
                      dynamic_load('lightgaussian.model', 'LightGaussian')),
    "chimerags": (ChimeraGSModelParams,
                  {
                        'distill': ChimeraGSDistillOptimizationParams,
                       'encode': ChimeraGSEncodeOptimizationParams
                       },
                  dynamic_load('chimerags.model', 'ChimeraGS')),
    "pup3dgs": (PUP3DGSModelParams, PUP3DGSOptimizationParams, dynamic_load('pup3dgs.model', 'PUP3DGS')),
    "speedy_splat": (SpeedySplatModelParams, SpeedySplatOptimizationParams, dynamic_load('speedy_splat.model', 'SpeedySplat')),
    "2dgs": (SurfelGSModelParams, SurfelGSOptimizationParams, dynamic_load('surfel_gs.model', 'SurfelGS')),
    "gsdraa": (GSDRAAModelParams, GSDRAAOptimizationParams, dynamic_load('gs_dr_aa.model', 'GSDRAA')),
    "trim3dgs": (Trim3DGSModelParams, Trim3DGSOptimizationParams, dynamic_load('trim3dgs.model', 'Trim3DGS')),
    "mesongs": (MesonGSModelParams, MesonGSOptimizationParams, dynamic_load('mesongs.model', 'MesonGS'))
}
