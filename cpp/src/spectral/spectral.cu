/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuml/cluster/spectral.hpp>

#include <sparse/coo.cuh>
#include <sparse/linalg/spectral.cuh>

namespace ML {

namespace Spectral {

void fit_embedding(const raft::handle_t &handle, int *rows, int *cols,
                   float *vals, int nnz, int n, int n_components, float *out) {
  raft::sparse::spectral::fit_embedding(handle, rows, cols, vals, nnz, n,
                                        n_components, out);
}
}  // namespace Spectral
}  // namespace ML
