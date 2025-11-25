from einops import rearrange
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from splatwizard.compression.entropy_model import Softmax
from .kmeans import kmeans


class ECVQ(nn.Module):
    def __init__(
            self,
            x_dim: int,
            cb_size: int,
            cb_dim: int,
            lmbda: float,
            rate_constrain: bool = True,
            lr: float = 2e-4,
            patch_size = 16384
    ):
        super().__init__()
        self.lr = lr
        self.lmbda = lmbda
        self.cb_size = cb_size
        self.cb_dim = cb_dim
        self._rate_constrain = rate_constrain
        self.patch_size = patch_size

        cb = x_dim // cb_dim
        assert x_dim == cb * cb_dim
        self.cb = cb
        self.codebook = nn.Parameter(
            torch.Tensor(cb, cb_size, cb_dim).normal_(0, 1. / math.sqrt(cb_dim)))
        self.logits = nn.Parameter(torch.zeros(cb, cb_size))

        # dynamic training settings
        self.rate_constrain = rate_constrain

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [{'params': self.parameters(), 'initial_lr': self.lr}],
            lr=self.lr,
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(self.trainer.max_steps * 0.8)],
            gamma=0.1,
            last_epoch=self.global_step
        )
        lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "step"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
    def forward(self, x, index_cache=None):
        if self.patch_size != 0:
            x_hat = []
            log2_prob = []
            x_index = []
            x_patched = x.split(self.patch_size, dim=0)
            index_cache_patched = index_cache.split(self.patch_size, dim=0) if index_cache is not None else [None] * len(x_patched)
            for patch_idx, b in enumerate(x_patched):
                x_hat_patched_patch, log2_prob_patch, x_index_patch = self.quant(b, index_cache=index_cache_patched[patch_idx])                    
                x_hat.append(x_hat_patched_patch)
                log2_prob.append(log2_prob_patch)
                x_index.append(x_index_patch)
            x_hat = torch.cat(x_hat, dim=0)
            log2_prob = torch.cat(log2_prob, dim=0)
            x_index = torch.cat(x_index, dim=0)
        else:
            x_hat, log2_prob, x_index = self.quant(x)

        bits = log2_prob.sum()
        return {
            "x_hat": x_hat,
            "bits": bits,
            "x_index": x_index
        }

    def quant(self, x, index_cache=None):
        """
        Args:
            x: (b, cb * cb_dim)
        Return:
            x_hat: (b, cb * cb_dim)
            log2_prob: (b, cb)
            index: (b, cb)
        """
        
        # quant_start_event = torch.cuda.Event(enable_timing=True)
        # quant_start_event.record()
        # torch.cuda.synchronize()  # Wait for the events to be recorded!
        
        x = rearrange(x, "b (cb cb_dim) -> b cb cb_dim", cb_dim=self.cb_dim)
        codebook = self.codebook
        log2_pmf = Softmax(self.logits).log_pmf() / (-math.log(2))  # cb, cb_size
        
        # quant_dist_event = torch.cuda.Event(enable_timing=True)
        # quant_dist_event.record()
        # torch.cuda.synchronize()  # Wait for the events to be recorded!
        
        if index_cache is None:
            # l2 distance
            x1 = rearrange(x, "b cb cb_dim -> cb b cb_dim")
            dist = torch.cdist(x1, codebook, p=2)
            dist = rearrange(dist, "cb b cb_size -> b cb cb_size")

            if self.rate_constrain:
                dist = dist + log2_pmf / self.lmbda

            index = dist.argmin(dim=-1, keepdim=True)  # b, cb, 1
        else:
            index = rearrange(index_cache, "b cb -> b cb 1")
        # one_hot = torch.zeros_like(dist).scatter_(-1, index, 1.0)  # b, cb, cb_size
        
        # quant_index_event = torch.cuda.Event(enable_timing=True)
        # quant_index_event.record()
        # torch.cuda.synchronize()  # Wait for the events to be recorded!
        
        x_hat = rearrange(torch.index_select(codebook, 1, index.squeeze()), "cb b cb_dim -> b cb cb_dim")
        log2_prob = rearrange(torch.index_select(log2_pmf, 1, index.squeeze()), "cb b -> b cb")

        x_hat = rearrange(x_hat, "b cb cb_dim -> b (cb cb_dim)", cb_dim=self.cb_dim)
        index = rearrange(index, "b cb 1 -> b cb")
        
        # quant_end_event = torch.cuda.Event(enable_timing=True)
        # quant_end_event.record()
        # torch.cuda.synchronize()  # Wait for the events to be recorded!
        
        # print('Quant start time', quant_start_event.elapsed_time(quant_dist_event))
        # print('Quant dist time', quant_dist_event.elapsed_time(quant_index_event))
        # print('Quant index time', quant_index_event.elapsed_time(quant_end_event))
        
        return x_hat, log2_prob, index
    
    
    @torch.no_grad()
    def kmeans_initialize(self, kmeans_init_tensor):
        assert kmeans_init_tensor.shape[1] == self.cb * self.cb_dim
        x = rearrange(kmeans_init_tensor, "b (cb cb_dim) -> cb b cb_dim", cb_dim=self.cb_dim)
        means, bins = kmeans(x, num_clusters=self.cb_size, patch_size=self.patch_size)
        assert means.shape == self.codebook.shape
        assert bins.shape == self.logits.shape
        self.codebook.data = means.to(self.codebook.device)
        self.logits.data = torch.log(F.normalize(bins.to(torch.float32), dim=-1)).to(self.logits.device)

    def codebook_info(self, prob_threshold):
        threshold = math.log(prob_threshold)
        log_pmf = Softmax(self.logits).log_pmf()
        mask_dead = log_pmf.le(threshold)
        mask_live = ~mask_dead
        num_live = mask_live.int().sum(-1)
        return num_live, mask_live

    def print_codebook_info(self, prob_threshold=5e-9):
        num_live, _ = self.codebook_info(prob_threshold)

    def reactivate_codeword(self, prob_threshold=1e-6):
        log_pmf = Softmax(self.logits).log_pmf()
        codebook = self.codebook.detach()
        logits = self.logits.detach()

        cb, cb_size, cb_dim = codebook.shape
        num_live, mask_live = self.codebook_info(prob_threshold)
        num_dead = cb_size - num_live

        for icb in range(cb):
            if num_dead[icb] == 0:
                continue
            mask = mask_live[icb]
            pmf = log_pmf[icb][mask].exp()
            idx = pmf.multinomial(num_dead[icb], replacement=True)
            disturb = torch.normal(0, 1e-4, size=(num_dead[icb], cb_dim))
            disturb = disturb.to(codebook.device)
            codebook[icb][~mask] = codebook[icb][mask][idx] + disturb
            logits[icb][~mask] = logits[icb][mask][idx]

        self.codebook.data = codebook
        self.logits.data = logits

        num_live_new, _ = self.codebook_info(prob_threshold)
        print(f'number of reactivated code_words (p>{prob_threshold}): '
              f'{sorted(num_live.view(-1).tolist())}'
              f' -> {sorted(num_live_new.view(-1).tolist())}')


def get_vq_cfg(opt):
    vq_cfg = {
        'lmbda': {
            'scale': opt.vq_scale_lmbda,
            'rot': opt.vq_rot_lmbda,
            'dc': opt.vq_dc_lmbda,
            'sh1': opt.vq_sh1_lmbda,
            'sh2': opt.vq_sh2_lmbda,
            'sh3': opt.vq_sh3_lmbda
        },
        'cb_size': {
            'scale': opt.vq_scale_cbsize,
            'rot': opt.vq_rot_cbsize,
            'dc': opt.vq_dc_cbsize,
            'sh1': opt.vq_sh1_cbsize,
            'sh2': opt.vq_sh2_cbsize,
            'sh3': opt.vq_sh3_cbsize
        },
        'cb_lr': opt.vq_cb_lr,
        'logits_lr': opt.vq_logits_lr,
        'patch_size': opt.vq_patch_size
    }
    vq_cfg['keys'] = vq_cfg['lmbda'].keys()
    
    return vq_cfg