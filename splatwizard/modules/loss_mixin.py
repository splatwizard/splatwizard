import torch

from splatwizard._cmod.fused_ssim import fused_ssim
from splatwizard.config import OptimizationParams
from splatwizard.metrics.loss_utils import l1_func, ssim_func
from splatwizard.modules.dataclass import LossPack, RenderResult
from splatwizard.profiler import profile


class LossMixin:
    def loss_func(self, viewpoint_cam, render_result: RenderResult, opt: OptimizationParams) -> (torch.Tensor, LossPack):
        gt_image = viewpoint_cam.original_image
        Ll1 = l1_func(render_result.rendered_image, gt_image)

        if opt.use_fused_ssim:
            ssim_value = fused_ssim(render_result.rendered_image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim_func(render_result.rendered_image, gt_image)

        ssim_loss = (1.0 - ssim_value)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        loss_pack = LossPack(
            l1_loss=Ll1,
            ssim_loss=ssim_loss,
            loss=loss
        )

        return loss, loss_pack
