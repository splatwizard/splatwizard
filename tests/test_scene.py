import pytest
import os

from splatwizard.scene import Scene

from splatwizard.config import PipelineParams


def test_load_colmap():
    dataset_path = os.environ['SW_DATASET']
    ppl = PipelineParams(source_path=dataset_path, debug=True)

    scene = Scene(ppl, )


    # print(dataset_path)