import inspect
import random
import re

import torch
import numpy as np
def wrap_str(*args):
    return ' '.join([str(a) for a in args])


def safe_state(seed=0, silent=None):
    # old_f = sys.stdout
    # class F:
    #     def __init__(self, silent):
    #         self.silent = silent
    #
    #     def write(self, x):
    #         if not self.silent:
    #             if x.endswith("\n"):
    #                 old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
    #             else:
    #                 old_f.write(x)
    #
    #     def flush(self):
    #         old_f.flush()
    #
    # sys.stdout = F(silent)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.set_device(torch.device("cuda:0"))




def check_tensor(value: torch.Tensor):
    torch.set_printoptions(8)
    stacks = inspect.stack()
    function_name = stacks[0].function
    s = inspect.stack()[1].code_context[0]
    arg_name = re.findall(f'.*{function_name}\((.*?)\)', s)
    nan_count = torch.isnan(value).sum()
    max_weight = value.max()
    min_weight = value.min()
    mean_weight = value.double().mean()
    std_weight = value.double().std()
    non_zero = (value != 0).sum()
    print(arg_name,
          '\t\t[SHAPE]', value.size(),
          '\t[DTYPE]', value.dtype,
          '\t[MEAN]', mean_weight, '\t[STD]', std_weight,
          '\t[NaN]', nan_count,
          '\t[MAX]', max_weight, '\t[MIN]', min_weight, '\t[NON ZERO]', non_zero / value.numel()
          )