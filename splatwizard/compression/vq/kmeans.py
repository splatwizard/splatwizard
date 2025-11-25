import torch
from einops import rearrange, repeat
import torch.nn.functional as F

def patched_bincount(x, *, minlength):
    patch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(patch, minlength, dtype = dtype, device = device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target

def patched_dist(x: torch.Tensor, codebook, patch_size=16384):
    x_patched = x.split(patch_size, dim=1)
    
    dist = []
    for b in x_patched:
        d = torch.cdist(b, codebook, 2)
        dist.append(d)
    dist = torch.cat(dist, dim=1)
    
    return dist

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]

def patched_sample_vectors(samples, num):
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim = 0)], dim = 0)

def noop(*args, **kwargs):
    pass

def kmeans(
    samples,
    num_clusters,
    num_iters = 10,
    use_cosine_sim = False,
    sample_fn = patched_sample_vectors,
    all_reduce_fn = noop,
    patch_size=16384
):
    num_codebooks, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device

    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        buckets = []
        samples_patched = samples.split(patch_size, dim=1)
        for b in samples_patched:
            if use_cosine_sim:
                dists = b @ rearrange(means, 'h n d -> h d n')
            else:
                dists = -patched_dist(b, means, patch_size)
                # dists = -torch.cdist(samples, means, p = 2, compute_mode='use_mm_for_euclid_dist_if_necessary')

            buckets.append(torch.argmax(dists, dim = -1))
        buckets = torch.cat(buckets, dim=1)
        bins = patched_bincount(buckets, minlength = num_clusters)
        all_reduce_fn(bins)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype = dtype)

        new_means.scatter_add_(1, repeat(buckets, 'h n -> h n d', d = dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, '... -> ... 1')
        all_reduce_fn(new_means)

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(
            rearrange(zero_mask, '... -> ... 1'),
            means,
            new_means
        )

    return means, bins