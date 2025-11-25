import sys

from splatwizard.main import main
from splatwizard.model_zoo import register_model
from splatwizard.model_zoo.gs.model import GSModel, GSModelParams, GSOptimizationParams
from splatwizard.modules.render_mixin import FlashGSRenderMixin


class FlashGS(FlashGSRenderMixin, GSModel):
    def __init__(self, model_params):
        GSModel.__init__(self, model_params)
        FlashGSRenderMixin.__init__(self)


if __name__ == "__main__":
    register_model('flashgs', GSModelParams, GSOptimizationParams, FlashGS)
    sys.exit(main())
