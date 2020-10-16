#
# Copyright (c) 2019, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from cuml.datasets.arima import make_arima
from cuml.datasets.blobs import make_blobs
from cuml.datasets.regression import make_regression
from cuml.datasets.classification import make_classification

__all__ = [
    "make_arima",
    "make_blobs",
    "make_classification",
    "make_regression",
]
