#
# Copyright (c) 2020, NVIDIA CORPORATION.
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

# import typing

# TODO: (MDD) typing.runtime_checkable is only in python 3.8. Need to determine
# if this is needed. For now, will work around it for 3.7
# @typing.runtime_checkable
# class ArrayOutputable(typing.Protocol):
#     """
#     Protocol class used to determine if a class can be converted to device
#     array-like objects
#     """
#     def to_output(self, output_type='cupy', output_dtype=None):
#         ...