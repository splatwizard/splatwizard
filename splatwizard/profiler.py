import time

import torch.cuda


class profile:
    def __init__(self, skip=False):
        self.start_time = None
        self.end_time = None
        self.duration = 0
        self.skip = skip
        self.peak_memory_allocated = 0
        self.peak_memory_reserved = 0

    def __enter__(self):
        if not self.skip:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            self.start_time = time.time()
        return self

    def __exit__(self, *exc_info):
        if self.skip:
            return
        torch.cuda.synchronize()
        # Track peak memory usage (in bytes)
        self.peak_memory_allocated = torch.cuda.max_memory_allocated()
        self.peak_memory_reserved = torch.cuda.max_memory_reserved()
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
