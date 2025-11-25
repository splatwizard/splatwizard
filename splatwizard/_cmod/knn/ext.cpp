/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include "knn/knn.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("knn_points_idx", &KNearestNeighborIdx);
  m.def("knn_points_backward", &KNearestNeighborBackward);
}