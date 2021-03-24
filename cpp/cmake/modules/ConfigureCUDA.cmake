#=============================================================================
# Copyright (c) 2018-2021, NVIDIA CORPORATION.
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
#=============================================================================

# Find the CUDAToolkit
find_package(CUDAToolkit REQUIRED)

if((CUDAToolkit_VERSION EQUAL 11) OR (CUDAToolkit_VERSION GREATER 11))
    set(CUB_IS_PART_OF_CTK ON)
else()
    message(STATUS "CUB NOT IN CTK")
    set(CUB_IS_PART_OF_CTK OFF)
endif()

# Auto-detect available GPU compute architectures
include(${CUML_SOURCE_DIR}/cmake/modules/SetGPUArchs.cmake)
message(STATUS "CUML: Building for GPU architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Must come after find_package(CUDAToolkit) because we symlink
# ccache as a compiler front-end for nvcc in gpuCI CPU builds.
# Must also come after we detect and potentially rewrite
# CMAKE_CUDA_ARCHITECTURES
enable_language(CUDA)
