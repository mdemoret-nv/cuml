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

import multiprocessing
import operator

import cupy as cp
import numpy as np
from cudf.core import Buffer
from cudf.core import DataFrame
from cudf.core import Series
from cuml.common.array_outputable import ArrayOutputable
from cuml.common.memory_utils import _get_size_from_shape
from cuml.common.memory_utils import _order_to_strides
from cuml.common.memory_utils import _strides_to_order
from cuml.common.memory_utils import class_with_cupy_rmm
from cuml.common.memory_utils import with_cupy_rmm
from numba import cuda
from rmm import DeviceBuffer

_array_manager = multiprocessing.Manager()

_to_output_counts = _array_manager.dict()
_from_array_counts = _array_manager.dict()
_malloc_nbytes = _array_manager.Value(int, 0)
_malloc_count = _array_manager.Value(int, 0)

def _increment_to_output(output_type: str):
    _to_output_counts[output_type] = _to_output_counts.setdefault(output_type, 0) + 1

def _increment_from_array(output_type: str):
    _from_array_counts[output_type] = _from_array_counts.setdefault(output_type, 0) + 1

def _increment_malloc(nbytes: int):
    _malloc_nbytes.set(_malloc_nbytes.get() + nbytes)
    _malloc_count.set(_malloc_count.get() + 1)

@class_with_cupy_rmm(ignore_pattern=["serialize"])
class CumlArray(Buffer, ArrayOutputable):

    """
    Array represents an abstracted array allocation. It can be instantiated by
    itself, creating an rmm.DeviceBuffer underneath, or can be instantiated by
    ``__cuda_array_interface__`` or ``__array_interface__`` compliant arrays,
    in which case it'll keep a reference to that data underneath. Also can be
    created from a pointer, specifying the characteristics of the array, in
    that case the owner of the data referred to by the pointer should be
    specified explicitly.

    Parameters
    ----------

    data : rmm.DeviceBuffer, cudf.Buffer, array_like, int, bytes, bytearray or\
           memoryview
        An array-like object or integer representing a
        device or host pointer to pre-allocated memory.
    owner : object, optional
        Python object to which the lifetime of the memory
        allocation is tied. If provided, a reference to this
        object is kept in this Buffer.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy or cupy data type.
    shape : int or tuple of ints, optional
        Shape of created array.
    order: string, optional
        Whether to create a F-major or C-major array.

    Attributes
    ----------

    ptr : int
        Pointer to the data
    size : int
        Size of the array data in bytes
    _owner : Python Object
        Object that owns the data of the array
    shape : tuple of ints
        Shape of the array
    order : {'F', 'C'}
        'F' or 'C' to indicate Fortran-major or C-major order of the array
    strides : tuple of ints
        Strides of the data
    __cuda_array_interface__ : dictionary
        ``__cuda_array_interface__`` to interop with other libraries.

    Notes
    -----

    cuml Array is not meant as an end-user array library. It is meant for
    cuML/RAPIDS developer consumption. Therefore it contains the minimum
    functionality. Its functionality is hidden by base.pyx to provide
    automatic output format conversion so that the users see the important
    attributes in whatever format they prefer.

    Todo: support cuda streams in the constructor. See:
    https://github.com/rapidsai/cuml/issues/1712
    https://github.com/rapidsai/cuml/pull/1396

    """

    def __init__(self, data=None, owner=None, dtype=None, shape=None,
                 order=None):

        # Checks of parameters
        memview_construction = False
        if data is None:
            raise TypeError("To create an empty Array, use the class method" +
                            " Array.empty().")
        elif isinstance(data, memoryview):
            data = np.asarray(data)
            memview_construction = True

        if dtype is not None:
            dtype = np.dtype(dtype)

        if _check_low_level_type(data):
            if dtype is None or shape is None or order is None:
                raise TypeError("Need to specify dtype, shape and order when" +
                                " creating an Array from {}."
                                .format(type(data)))
            detailed_construction = True
        elif dtype is not None and shape is not None and order is not None:
            detailed_construction = True
        else:
            # Catch a likely developer error if CumlArray is created
            # incorrectly
            assert dtype is None and shape is None and order is None, \
                ("Creating array from array-like object. The arguments "
                 "`dtype`, `shape` and `order` should be `None`.")

            detailed_construction = False

        ary_interface = False

        # Base class (Buffer) constructor call
        size, shape = _get_size_from_shape(shape, dtype)

        if not memview_construction and not detailed_construction:
            # Convert to cupy array and manually specify the ptr, size and
            # owner. This is to avoid the restriction on Buffer that requires
            # all data be u8
            cupy_data = cp.asarray(data)
            flattened_data = cupy_data.data.ptr

            # Size for Buffer is not the same as for cupy. Use nbytes
            size = cupy_data.nbytes
            owner = cupy_data if cupy_data.flags.owndata else data
        else:
            flattened_data = data

        super(CumlArray, self).__init__(data=flattened_data,
                                        owner=owner,
                                        size=size)

        # Post processing of meta data
        if detailed_construction:
            self.shape = shape
            self.dtype = dtype
            self.order = order
            self.strides = _order_to_strides(order, shape, dtype)

        elif hasattr(data, "__array_interface__"):
            ary_interface = data.__array_interface__

        elif hasattr(data, "__cuda_array_interface__"):
            ary_interface = data.__cuda_array_interface__

        else:
            raise TypeError("Unrecognized data type: %s" % str(type(data)))

        if ary_interface:
            self.shape = ary_interface['shape']
            self.dtype = np.dtype(ary_interface['typestr'])
            if ary_interface.get('strides', None) is None:
                self.order = 'C'
                self.strides = _order_to_strides(self.order, self.shape,
                                                 self.dtype)
            else:
                self.strides = ary_interface['strides']
                self.order = _strides_to_order(self.strides, self.dtype)

    @with_cupy_rmm
    def __getitem__(self, slice):
        return CumlArray(data=cp.asarray(self).__getitem__(slice))

    def __setitem__(self, slice, value):
        cp.asarray(self).__setitem__(slice, value)

    def __len__(self):
        return self.shape[0]

    def _operator_overload(self, other, fn):
        return CumlArray(fn(self.to_output('cupy'), other))

    def __add__(self, other):
        return self._operator_overload(other, operator.add)

    def __sub__(self, other):
        return self._operator_overload(other, operator.sub)

    @property
    def __cuda_array_interface__(self):
        output = {
            "shape": self.shape,
            "strides": self.strides,
            "typestr": self.dtype.str,
            "data": (self.ptr, False),
            "version": 2,
        }
        return output

    def item(self):
        return cp.asarray(self).item()

    def to_output(self, output_type='cupy', output_dtype=None):
        """
        Convert array to output format

        Parameters
        ----------
        output_type : string
            Format to convert the array to. Acceptable formats are:

            - 'cupy' - to cupy array
            - 'numpy' - to numpy (host) array
            - 'numba' - to numba device array
            - 'dataframe' - to cuDF DataFrame
            - 'series' - to cuDF Series
            - 'cudf' - to cuDF Series if array is single dimensional, to
               DataFrame otherwise

        output_dtype : string, optional
            Optionally cast the array to a specified dtype, creating
            a copy if necessary.

        """
        if output_dtype is None:
            output_dtype = self.dtype

        # check to translate cudf to actual type converted
        if output_type == 'cudf':
            if len(self.shape) == 1:
                output_type = 'series'
            elif self.shape[1] == 1:
                output_type = 'series'
            else:
                output_type = 'dataframe'

        assert output_type != "mirror"

        _increment_to_output(output_type)

        if output_type == 'cupy':
            return cp.asarray(self, dtype=output_dtype)

        elif output_type == 'numba':
            return cuda.as_cuda_array(cp.asarray(self, dtype=output_dtype))

        elif output_type == 'numpy':
            return cp.asnumpy(
                cp.asarray(self, dtype=output_dtype), order=self.order
            )

        elif output_type == 'dataframe':
            if self.dtype not in [np.uint8, np.uint16, np.uint32,
                                  np.uint64, np.float16]:
                mat = cp.asarray(self, dtype=output_dtype)
                if len(mat.shape) == 1:
                    mat = mat.reshape(mat.shape[0], 1)
                return DataFrame(mat)
            else:
                raise ValueError('cuDF unsupported Array dtype')

        elif output_type == 'series':
            # check needed in case output_type was passed as 'series'
            # directly instead of as 'cudf'
            if len(self.shape) == 1:
                if self.dtype not in [np.uint8, np.uint16, np.uint32,
                                      np.uint64, np.float16]:
                    return Series(self, dtype=output_dtype)
                else:
                    raise ValueError('cuDF unsupported Array dtype')
            elif self.shape[1] > 1:
                raise ValueError('Only single dimensional arrays can be \
                                 transformed to cuDF Series. ')
            else:
                if self.dtype not in [np.uint8, np.uint16, np.uint32,
                                      np.uint64, np.float16]:
                    return Series(self, dtype=output_dtype)
                else:
                    raise ValueError('cuDF unsupported Array dtype')

    def serialize(self):
        header, frames = super(CumlArray, self).serialize()
        header["constructor-kwargs"] = {
            "dtype": self.dtype.str,
            "shape": self.shape,
            "order": self.order,
        }
        frames = [Buffer(f) for f in frames]
        return header, frames

    @classmethod
    def empty(cls, shape, dtype, order='F'):
        """
        Create an empty Array with an allocated but uninitialized DeviceBuffer

        Parameters
        ----------
        dtype : data-type, optional
            Any object that can be interpreted as a numpy or cupy data type.
        shape : int or tuple of ints, optional
            Shape of created array.
        order: string, optional
            Whether to create a F-major or C-major array.
        """

        # size, _ = _get_size_from_shape(shape, dtype)
        # dbuf = DeviceBuffer(size=size)
        # return CumlArray(data=dbuf, shape=shape, dtype=dtype, order=order)
        return CumlArray(cp.empty(shape, dtype, order))

    @classmethod
    def full(cls, shape, value, dtype, order='F'):
        """
        Create an Array with an allocated DeviceBuffer initialized to value.

        Parameters
        ----------
        dtype : data-type, optional
            Any object that can be interpreted as a numpy or cupy data type.
        shape : int or tuple of ints, optional
            Shape of created array.
        order: string, optional
            Whether to create a F-major or C-major array.
        """
        # size, _ = _get_size_from_shape(shape, dtype)
        # dbuf = DeviceBuffer(size=size)
        # cp.asarray(dbuf).view(dtype=dtype).fill(value)
        # return CumlArray(data=dbuf, shape=shape, dtype=dtype,
        #                  order=order)
        return CumlArray(cp.full(shape, value, dtype, order))

    @classmethod
    def zeros(cls, shape, dtype='float32', order='F'):
        """
        Create an Array with an allocated DeviceBuffer initialized to zeros.

        Parameters
        ----------
        dtype : data-type, optional
            Any object that can be interpreted as a numpy or cupy data type.
        shape : int or tuple of ints, optional
            Shape of created array.
        order: string, optional
            Whether to create a F-major or C-major array.
        """
        return CumlArray.full(value=0, shape=shape, dtype=dtype, order=order)

    @classmethod
    def ones(cls, shape, dtype='float32', order='F'):
        """
        Create an Array with an allocated DeviceBuffer initialized to zeros.

        Parameters
        ----------
        dtype : data-type, optional
            Any object that can be interpreted as a numpy or cupy data type.
        shape : int or tuple of ints, optional
            Shape of created array.
        order: string, optional
            Whether to create a F-major or C-major array.
        """
        return CumlArray.full(value=1, shape=shape, dtype=dtype, order=order)


def _check_low_level_type(data):
    if isinstance(data, CumlArray):
        return False
    elif not (
        hasattr(data, "__array_interface__")
        or hasattr(data, "__cuda_array_interface__")
    ) or isinstance(data, (DeviceBuffer, Buffer)):
        return True
    else:
        return False
