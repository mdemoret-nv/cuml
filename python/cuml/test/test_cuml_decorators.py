import contextlib
import typing
from collections import deque

import numpy as np
import cupy as cp

import cuml
import cudf
from cuml.common.input_utils import input_to_cuml_array, determine_array_type, determine_array_dtype, unsupported_cudf_dtypes
import cuml.internals
import pytest
from cuml.common.array import CumlArray
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.test.utils import array_equal

test_input_types = ['numpy', 'numba', 'cupy', 'cudf']

test_output_types_str = ['numpy', 'numba', 'cupy', 'cudf']

test_output_types = {
    'numpy': np.ndarray,
    'cupy': cp.ndarray,
    'numba': None,
    'series': cudf.Series,
    'dataframe': cudf.DataFrame,
    'cudf': None
}

test_dtypes_all = [
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    "float",
    "float32",
    "double",
    "float64",
    "int8",
    "short",
    "int16",
    "int",
    "int32",
    "long",
    "int64",
]

test_dtypes_output = [
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64
]

test_dtypes_short = [
    np.uint8,
    np.float16,
    np.int32,
    np.float64,
]

test_shapes = [10, (10, 1), (10, 5), (1, 10)]

test_slices = [0, 5, 'left', 'right', 'both', 'bool_op']


class TestEstimator(cuml.Base):

    input_any_ = CumlArrayDescriptor()
    input_cuml_ = CumlArrayDescriptor()
    input_numpy_ = CumlArrayDescriptor()
    input_numba_ = CumlArrayDescriptor()
    input_dataframe_ = CumlArrayDescriptor()
    input_cupy_ = CumlArrayDescriptor()
    input_series_ = CumlArrayDescriptor()

    def _set_input(self, X):
        self.input_any_ = X

    @cuml.internals.api_base_return_any()
    def store_input(self, X):
        self.input_any_ = X

    @cuml.internals.api_return_any()
    def get_input(self):
        return self.input_any_

    # === Standard Functions ===
    def fit(self, X, convert_dtype=True) -> "TestEstimator":

        return self

    def predict(self, X, convert_dtype=True) -> CumlArray:

        return X

    def transform(self, X, convert_dtype=False) -> CumlArray:

        pass

    def fit_transform(self, X, y=None) -> CumlArray:

        return self.fit(X).transform(X)

    # === Auto Wrap By Return Functions ===
    def autowrap_return_float(self) -> float:

        pass

    def autowrap_return_self(self) -> float:

        pass

    def autowrap_return_cumlarray(self) -> float:

        pass

    def autowrap_return_union_cumlarray(self) -> float:

        pass

    def autowrap_return_tuple_cumlarray(self) -> float:

        pass

    def autowrap_return_list_cumlarray(self) -> float:

        pass

    def autowrap_return_dict_cumlarray(self) -> float:

        pass

    # === Explicit Return Functions ===
    def explicit_return_float(self):

        pass

    def explicit_return_self(self):

        pass

    def explicit_return_cumlarray(self):

        pass

    def explicit_return_union_cumlarray(self):

        pass

    def explicit_return_tuple_cumlarray(self):

        pass

    def explicit_return_list_cumlarray(self):

        pass

    def explicit_return_dict_cumlarray(self):

        pass


def array_identical(a, b):

    cupy_a = input_to_cuml_array(a, order="K").array
    cupy_b = input_to_cuml_array(b, order="K").array

    if len(a) == 0 and len(b) == 0:
        return True

    if (cupy_a.shape != cupy_b.shape):
        return False

    if (cupy_a.dtype != cupy_b.dtype):
        return False

    if (cupy_a.order != cupy_b.order):
        return False

    return cp.all(cp.asarray(cupy_a) == cp.asarray(cupy_b)).item()


def create_input(input_type, input_dtype, input_shape, input_order):
    float_dtypes = [np.float16, np.float32, np.float64]
    if input_dtype in float_dtypes:
        rand_ary = cp.random.random(input_shape)
    else:
        rand_ary = cp.random.randint(100, size=input_shape)

    rand_ary = cp.array(rand_ary, dtype=input_dtype, order=input_order)

    cuml_ary = CumlArray(rand_ary)

    return cuml_ary.to_output(input_type)


def create_output(X_in, output_type):

    cuml_ary_tuple = input_to_cuml_array(X_in, order="K")

    return cuml_ary_tuple.array.to_output(output_type)


@pytest.mark.parametrize('input_type', test_input_types)
@pytest.mark.parametrize('input_dtype', [np.float32, np.int16])
@pytest.mark.parametrize('input_shape', [10, (10, 5)])
@pytest.mark.parametrize('output_type', test_output_types_str)
def test_dec_input_output(input_type, input_dtype, input_shape, output_type):

    if (input_type == "cudf" or output_type == "cudf"):
        if (input_dtype in unsupported_cudf_dtypes):
            pytest.skip("Unsupported cudf combination")

    X_in = create_input(input_type, input_dtype, input_shape, "C")
    X_out = create_output(X_in, output_type)

    # Test with output_type="input"
    est = TestEstimator(output_type="input")

    est.store_input(X_in)

    # Test is was stored internally correctly
    assert X_in is est.get_input()

    assert est.__dict__["input_any_"].input_type == input_type

    # Check the current type matches input type
    assert determine_array_type(est.input_any_) == input_type

    assert array_identical(est.input_any_, X_in)

    # Switch output type and check type and equality
    with cuml.using_output_type(output_type):

        assert determine_array_type(est.input_any_) == output_type

        assert array_identical(est.input_any_, X_out)

    # Now Test with output_type=output_type
    est = TestEstimator(output_type=output_type)

    est.store_input(X_in)

    # Check the current type matches output type
    assert determine_array_type(est.input_any_) == output_type

    assert array_identical(est.input_any_, X_out)

    with cuml.using_output_type("input"):

        assert determine_array_type(est.input_any_) == input_type

        assert array_identical(est.input_any_, X_in)


@pytest.mark.parametrize('input_type', test_input_types)
@pytest.mark.parametrize('input_dtype', [np.float32, np.int16])
@pytest.mark.parametrize('input_shape', test_shapes)
def test_auto_fit(input_type, input_dtype, input_shape):
    """
    Test autowrapping on fit that will set output_type, and n_features
    """
    X_in = create_input(input_type, input_dtype, input_shape, "C")

    # Test with output_type="input"
    est = TestEstimator()

    est.fit(X_in)

    def calc_n_features(shape):
        if (isinstance(shape, tuple) and len(shape) >= 1):

            # When cudf and shape[1] is used, a series is created which will remove the last shape
            if (input_type == "cudf" and shape[1] == 1):
                return None

            return shape[1]

        return None

    assert est._input_type == input_type
    assert est.target_dtype is None
    assert est.n_features_in_ == calc_n_features(input_shape)


@pytest.mark.parametrize('input_type', test_input_types)
@pytest.mark.parametrize('base_output_type', test_input_types)
@pytest.mark.parametrize('global_output_type',
                         test_output_types_str + ["input", None])
def test_auto_predict(input_type, base_output_type, global_output_type):
    """
    Test autowrapping on predict that will set target_type
    """
    X_in = create_input(input_type, np.float32, (10, 10), "F")

    # Test with output_type="input"
    est = TestEstimator()

    # With cuml.global_output_type == None, this should return the input type
    X_out = est.predict(X_in)

    assert determine_array_type(X_out) == input_type

    assert array_identical(X_in, X_out)

    # Test with output_type=base_output_type
    est = TestEstimator(output_type=base_output_type)

    # With cuml.global_output_type == None, this should return the base_output_type
    X_out = est.predict(X_in)

    assert determine_array_type(X_out) == base_output_type

    assert array_identical(X_in, X_out)

    # Test with global_output_type, should return global_output_type
    with cuml.using_output_type(global_output_type):
        X_out = est.predict(X_in)

        target_output_type = global_output_type

        if (target_output_type is None or target_output_type == "input"):
            target_output_type = base_output_type

        if (target_output_type == "input"):
            target_output_type = input_type

        assert determine_array_type(X_out) == target_output_type

        assert array_identical(X_in, X_out)


@pytest.mark.parametrize('input_arg', ["X", "y", "bad", None])
@pytest.mark.parametrize('target_arg', ["X", "y", "bad", None])
@pytest.mark.parametrize('skip_get_output_type', [True, False])
@pytest.mark.parametrize('skip_get_output_dtype', [True, False])
def test_return_array(input_arg: str,
                      target_arg: str,
                      skip_get_output_type: bool,
                      skip_get_output_dtype: bool):
    """
    Test autowrapping on predict that will set target_type
    """

    input_type_X = "numpy"
    input_dtype_X = np.float64

    input_type_Y = "cupy"
    input_dtype_Y = np.int32

    inner_type = "numba"
    inner_dtype = np.float16

    X_in = create_input(input_type_X, input_dtype_X, (10, 10), "F")
    Y_in = create_input(input_type_Y, input_dtype_Y, (10, 10), "F")

    def test_func(X, y):

        if (skip_get_output_type):
            cuml.internals.set_api_output_type(inner_type)

        if (skip_get_output_dtype):
            cuml.internals.set_api_output_dtype(inner_dtype)

        return X

    if (input_arg == "bad" or target_arg == "bad"):
        pytest.xfail("Expected error with bad arg name")

    test_func = cuml.internals.api_return_array(
        input_arg=input_arg,
        target_arg=target_arg,
        skip_get_output_type=skip_get_output_type,
        skip_get_output_dtype=skip_get_output_dtype)(test_func)

    X_out = test_func(X=X_in, y=Y_in)

    target_type = None
    target_dtype = None

    if (skip_get_output_type):
        target_type = inner_type
    else:
        if (input_arg == "y"):
            target_type = input_type_Y
        else:
            target_type = input_type_X

    if (skip_get_output_dtype):
        target_dtype = inner_dtype
    else:
        if (target_arg == "X"):
            target_dtype = input_dtype_X
        else:
            target_dtype = input_dtype_Y

    assert determine_array_type(X_out) == target_type

    assert determine_array_dtype(X_out) == target_dtype