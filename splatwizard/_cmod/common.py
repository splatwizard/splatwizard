import os

import torch


try:
    arch = os.environ['TORCH_CUDA_ARCH_LIST']
except KeyError:
    cap_ver = torch.cuda.get_device_capability()
    # print('set architecture to {}'.format(cap_ver))
    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{cap_ver[0]}.{cap_ver[1]}'


from torch.utils.cpp_extension import load