import os
import subprocess
import tempfile

import pandas as pd
import numpy as np
import pathlib

from plyfile import PlyElement, PlyData


def _decode_anchor(tmp_path: pathlib.Path, tmc3_path: pathlib.Path):
    ply_path = str(tmp_path / 'anchor_pc_decoded.ply')
    bin_path = str(tmp_path / 'anchor_compressed.drc')


    result = subprocess.run([
        str(tmc3_path.absolute()),
        '-c', './cfgs/decoder.cfg',
        f'--compressedStreamPath={bin_path}',
        f'--reconstructedDataPath={ply_path}'
    ]
    )
    assert result.returncode == 0
    plydata = PlyData.read(ply_path)

    anchor = np.stack((
        np.asarray(plydata.elements[0]["x"]),
        np.asarray(plydata.elements[0]["y"]),
        np.asarray(plydata.elements[0]["z"])
    ), axis=1).astype(np.float32)

    return anchor


def encode_anchor(q_anchor: np.array, tmc3_path: pathlib.Path):
    # print(q_anchor.shape)
    # print('estimated size', q_anchor.shape[0] * 3 * anchor_round_digits / 1024 / 1024 / 8, 'MB')

    origin_pc = pd.DataFrame({
        'x': q_anchor[:, 0],
        'y': q_anchor[:, 1],
        'z': q_anchor[:, 2],
        'order': range(q_anchor.shape[0])
    })

    origin_pc = origin_pc.sort_values(['x', 'y', 'z']).reset_index(drop=True)
    origin_order = origin_pc['order'].to_numpy()


    x = origin_pc['x'].to_numpy().reshape(-1, 1)
    y = origin_pc['y'].to_numpy().reshape(-1, 1)
    z = origin_pc['z'].to_numpy().reshape(-1, 1)

    # print(x.reshape(-1))
    # print(y.reshape(-1))
    # print(z.reshape(-1))
    dtype_full = [(attribute, 'f4') for attribute in ('x', 'y', 'z')]
    elements = np.empty(x.shape[0], dtype=dtype_full)
    attributes = np.concatenate((x, y, z), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')

    with tempfile.TemporaryDirectory() as dirname:
        dirname = pathlib.Path(dirname)
        ply_path = str(dirname / 'anchor_pc.ply')
        bin_path = str(dirname / 'anchor_compressed.drc')
        PlyData([el]).write(ply_path)

        result = subprocess.run([
            str(tmc3_path.absolute()),
            '-c', './cfgs/lossless_encoder.cfg',
            f'--uncompressedDataPath={ply_path}',
            f'--compressedStreamPath={bin_path}'
            ]
        )

        assert result.returncode == 0

        with open(bin_path, 'rb') as bin_file:
            bittream = bin_file.read()
        # anchor_bit = os.path.getsize(bin_path) * 8

        # print(result)

        # exit()

        # origin = pd.DataFrame({
        #     'x': q_anchor[:, 0],
        #     'y': q_anchor[:, 1],
        #     'z': q_anchor[:, 2]
        # })

        # origin = origin.sort_values(['x', 'y', 'z']).reset_index(drop=True)


        decoded_anchor = _decode_anchor(dirname, tmc3_path)

    decoded = pd.DataFrame({
        'x': decoded_anchor[:, 0],
        'y': decoded_anchor[:, 1],
        'z': decoded_anchor[:, 2],
        'order': range(decoded_anchor.shape[0])
    })

    decoded = decoded.sort_values(['x', 'y', 'z']).reset_index(drop=True)

    selection = origin_order[decoded['order'].to_numpy().argsort()]


    # t = q_anchor[selection]
    #
    # t =  np.abs((t -decoded_anchor)).sum()
    #
    # print(t)

    return selection, bittream



def decode_coordinate(bitstream, tmc3_path: pathlib.Path):
    with tempfile.TemporaryDirectory() as dirname:
        dirname = pathlib.Path(dirname)
        # ply_path = str(dirname / 'anchor_pc.ply')
        bin_path = str(dirname / 'anchor_compressed.drc')
        with open(bin_path, 'wb') as bin_file:
            bin_file.write(bitstream)

        xyz = _decode_anchor(dirname, tmc3_path)
        return xyz