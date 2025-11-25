import torch
from torch import nn
from torch.distributions import Uniform

anchor_round_digits = 16
Q_anchor = 1/(2 ** anchor_round_digits - 1)
# use_clamp = True
use_multiprocessor = False  # Always False plz. Not yet implemented for True.


class UniverseQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        #b = np.random.uniform(-1,1)
        b = 0
        uniform_distribution = Uniform(-0.5*torch.ones(x.size())
                                       * (2**b), 0.5*torch.ones(x.size())*(2**b)).sample().cuda()
        return torch.round(x+uniform_distribution)-uniform_distribution

    @staticmethod
    def backward(ctx, g):

        return g


class UniformQuantizer(nn.Module):
    def __init__(self, use_clamp=True):
        super().__init__()
        self.use_clamp = use_clamp

    def forward(self, x, Q, input_mean=None):
        if self.use_clamp:
            if input_mean is None:
                input_mean = x.mean()
            if not isinstance(Q, torch.Tensor):
                Q = torch.ones(1, device=x.device) * Q
            input_min = input_mean / Q.mean().detach() - 15_000
            input_max = input_mean / Q.mean().detach() + 15_000
            x = torch.clamp(x / Q, min=input_min.detach(), max=input_max.detach()) * Q

        x = x + torch.empty_like(x).uniform_(-0.5, 0.5) * Q
        return x


class STEQuantizer(nn.Module):
    def __init__(self, use_clamp=True):
        super().__init__()
        self.use_clamp = use_clamp

    def forward(self, input, Q, input_mean=None):
        return STE_multistep.apply(input, Q, input_mean, self.use_clamp)


class STEQuantizerFunc(torch.autograd.Function):
    """Actual quantization in the forward and set gradient to one ine the backward."""
    @staticmethod
    def forward(ctx, x: torch.Tensor, training: bool):
        # training is a dummy parameters used to have the same signature for both
        # quantizer forward functions.
        ctx.save_for_backward(x)
        y = torch.round(x)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        return grad_out, None  # No gradient with respect to <training> variable

# class STE_multistep(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, Q, input_mean=None):
#         Q_round = torch.round(input / Q)
#         Q_q = Q_round * Q
#         return Q_q
#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output, None


class STE_binary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = torch.clamp(input, min=-1, max=1)
        # out = torch.sign(input)
        p = (input >= 0) * (+1.0)
        n = (input < 0) * (-1.0)
        out = p + n
        return out
    @staticmethod
    def backward(ctx, grad_output):
        # mask: to ensure x belongs to (-1, 1)
        input, = ctx.saved_tensors
        i2 = input.clone().detach()
        i3 = torch.clamp(i2, -1, 1)
        mask = (i3 == i2) + 0.0
        return grad_output * mask


class STE_multistep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, Q, input_mean=None, use_clamp=True):
        if use_clamp:
            if input_mean is None:
                input_mean = input.mean()
            if not isinstance(Q, torch.Tensor):
                Q = torch.ones(1, device=input.device) * Q
            input_min = input_mean / Q.mean().detach() - 15_000
            input_max = input_mean / Q.mean().detach() + 15_000

            # print(int(input_min.detach()), int(input_max.detach()))
            input = torch.clamp(input / Q, min=int(input_min.detach()), max=int(input_max.detach())) * Q

        Q_round = torch.round(input / Q)
        Q_q = Q_round * Q

        return Q_q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class Quantize_anchor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, anchors, min_v, max_v):
        # if anchor_round_digits == 32:
            # return anchors
        # min_v = torch.min(anchors).detach()
        # max_v = torch.max(anchors).detach()
        # scales = 2 ** anchor_round_digits - 1
        interval = ((max_v - min_v) * Q_anchor + 1e-6)  # avoid 0, if max_v == min_v
        # quantized_v = (anchors - min_v) // interval
        quantized_v = torch.div(anchors - min_v, interval, rounding_mode='floor')
        quantized_v = torch.clamp(quantized_v, 0, 2 ** anchor_round_digits - 1)
        anchors_q = quantized_v * interval + min_v
        return anchors_q, quantized_v
    @staticmethod
    def backward(ctx, grad_output, tmp):  # tmp is for quantized_v:)
        return grad_output, None, None


class UniformNoiseQuantizer(torch.autograd.Function):
    """If training: use noise addition. Otherwise use actual quantization. Gradient is always one."""
    @staticmethod
    def forward(ctx, x: torch.Tensor, training: bool):
        ctx.save_for_backward(x)
        if training:
            y = x + (torch.rand_like(x) - 0.5) if training else torch.round(x)
        else:
            y = torch.round(x)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        return grad_out, None   # No gradient with respect to <training> variable