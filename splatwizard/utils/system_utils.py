from errno import EEXIST
from os import makedirs, path
import os
from pathlib import Path


def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def search_for_max_iteration(folder: Path):
    saved_iters = [int(fname.stem.split("_")[-1]) for fname in folder.iterdir()]
    return max(saved_iters)
