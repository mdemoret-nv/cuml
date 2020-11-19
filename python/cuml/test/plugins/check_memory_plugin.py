#
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
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
import datetime
import dataclasses
import typing
from copy import deepcopy
from functools import wraps

import _pytest.config
import _pytest.python
import _pytest.terminal
import cupy as cp
import pytest
import rmm
import rmm._lib
from pygments import highlight
from pygments.formatters.terminal import TerminalFormatter
from pygments.lexers.python import PythonLexer

_F = typing.TypeVar("_F", bound=typing.Callable[..., typing.Any])

# rmm.reinitialize(logging=True, log_file_name="test_log.txt")

global_tracked_mr = rmm.mr.TrackingMemoryResource(
    rmm.mr.get_current_device_resource())

# callback_mr = rmm.mr.CallbackResourceAdaptor(
#     rmm.mr.get_current_device_resource(),
#     None)

rmm.mr.set_current_device_resource(global_tracked_mr)


def _get_empty_alloc_info():
    return {
        "outstanding": 0,
        "peak": 0,
        "count": 0,
        "nbytes": 0,
        "streams": {
            0: 0,
        }
    }


@dataclasses.dataclass()
class ArrayAllocs:
    to_output_counts: dict = dataclasses.field(default_factory=dict)
    from_array_counts: dict = dataclasses.field(default_factory=dict)
    rmm_alloc_info: dict = dataclasses.field(
        default_factory=_get_empty_alloc_info)
    cupy_alloc_info: dict = dataclasses.field(
        default_factory=_get_empty_alloc_info)

    def increment_to_output(self, output_type: str):
        self.to_output_counts[output_type] = self.to_output_counts.setdefault(
            output_type, 0) + 1

    def increment_from_output(self, output_type: str):
        self.from_array_counts[
            output_type] = self.from_array_counts.setdefault(output_type,
                                                             0) + 1

    def reset(self):
        self.to_output_counts.clear()
        self.from_array_counts.clear()
        self.rmm_alloc_info = _get_empty_alloc_info()
        self.cupy_alloc_info = _get_empty_alloc_info()


test_memory_info: typing.Dict[str, ArrayAllocs] = {"total": ArrayAllocs()}

_total_array_alloc = ArrayAllocs()
_test_array_alloc = ArrayAllocs()


def _increment_to_output(output_type: str):

    _test_array_alloc.increment_to_output(output_type)
    _total_array_alloc.increment_to_output(output_type)


def _increment_from_array(output_type: str):

    _test_array_alloc.increment_from_output(output_type)
    _total_array_alloc.increment_from_output(output_type)


def _increment_malloc(nbytes: int):

    _total_array_alloc.cupy_alloc_info["count"] += 1
    _total_array_alloc.cupy_alloc_info["nbytes"] += nbytes

    _test_array_alloc.cupy_alloc_info["count"] += 1
    _test_array_alloc.cupy_alloc_info["nbytes"] += nbytes


# Set a bad cupy allocator that will fail if rmm.rmm_cupy_allocator is not used
def bad_allocator(nbytes):

    assert False, \
        "Using default cupy allocator instead of rmm.rmm_cupy_allocator"

    return None


saved_allocator = rmm.rmm_cupy_allocator


def counting_rmm_allocator(nbytes):

    _increment_malloc(nbytes)

    return saved_allocator(nbytes)


rmm.rmm_cupy_allocator = counting_rmm_allocator


def pytest_configure(config):
    cp.cuda.set_allocator(counting_rmm_allocator)

    import cuml.common.array
    import cuml.common.input_utils

    def tracked_to_output(func: _F) -> _F:
        @wraps(func)
        def inner(self, output_type='cupy', output_dtype=None):

            _increment_to_output(output_type)

            return func(self,
                        output_type=output_type,
                        output_dtype=output_dtype)

        return inner

    def tracked_input_to_cuml_array(func: _F) -> _F:
        @wraps(func)
        def inner(X, *args, **kwargs):

            _increment_from_array(
                cuml.common.input_utils.determine_array_type(X))

            return func(X, *args, **kwargs)

        return inner

    cuml.common.array.CumlArray.to_output = tracked_to_output(
        cuml.common.array.CumlArray.to_output)
    cuml.common.input_utils.input_to_cuml_array = tracked_input_to_cuml_array(
        cuml.common.input_utils.input_to_cuml_array)


@pytest.fixture(scope="function", autouse=True)
def cupy_allocator_fixture(request):

    # Disable creating cupy arrays
    # cp.cuda.set_allocator(bad_allocator)
    cp.cuda.set_allocator(counting_rmm_allocator)

    allocations = {}
    memory = {
        "outstanding": 0,
        "peak": 0,
        "count": 0,
        "nbytes": 0,
        "streams": {
            0: 0,
        }
    }

    def print_mr(is_alloc: bool, mem_ptr, n_bytes: int, stream_ptr):

        print("{},{},0x{:x},{},{}".format(
            datetime.datetime.now().strftime("%H:%M:%S.%f"),
            "allocate" if is_alloc else "free",
            mem_ptr,
            n_bytes,
            stream_ptr,
        ))

        if (stream_ptr not in memory["streams"]):
            memory["streams"][stream_ptr] = 0

        if (is_alloc):
            assert mem_ptr not in allocations
            allocations[mem_ptr] = n_bytes
            memory["outstanding"] += n_bytes
            memory["peak"] = max(memory["outstanding"], memory["peak"])
            memory["count"] += 1
            memory["nbytes"] += n_bytes
            memory["streams"][stream_ptr] += n_bytes
        else:
            # assert mem_ptr in allocations
            popped_nbytes = allocations.pop(mem_ptr, 0)
            memory["outstanding"] -= n_bytes if n_bytes > 0 else popped_nbytes

    # callback_mr.set_callback(print_mr)

    global_alloc_counts_pre = global_tracked_mr.get_allocation_counts()

    old_mr = rmm.mr.get_current_device_resource()

    try:
        tracked_mr = rmm.mr.TrackingMemoryResource(old_mr)

        rmm.mr.set_current_device_resource(tracked_mr)

        _test_array_alloc.reset()

        yield

        import gc

        gc.collect()

        temp_alloc_info = tracked_mr.get_allocation_counts()

        alloc_info = {
            "outstanding": temp_alloc_info["current_bytes"],
            "peak": temp_alloc_info["peak_bytes"],
            "count": temp_alloc_info["total_count"],
            "nbytes": temp_alloc_info["total_bytes"],
        }
    finally:
        rmm.mr.set_current_device_resource(old_mr)

    global_alloc_counts_post = global_tracked_mr.get_allocation_counts()

    # assert global_alloc_counts_pre[
    #     "current_count"] == global_alloc_counts_post["current_count"]
    # assert global_alloc_counts_pre[
    #     "current_bytes"] == global_alloc_counts_post["current_bytes"]

    # assert len(allocations) == 0

    # del memory["outstanding"]
    alloc_info["streams"] = {0: alloc_info["nbytes"]}

    test_array_info = deepcopy(_test_array_alloc)
    test_array_info.rmm_alloc_info = alloc_info

    test_memory_info[request.node.nodeid] = test_array_info

    test_memory_info["total"].rmm_alloc_info.update({
        "count":
            test_memory_info["total"].rmm_alloc_info["count"] +
            alloc_info["count"],
        "peak":
            max(test_memory_info["total"].rmm_alloc_info["peak"],
                alloc_info["peak"]),
        "nbytes":
            test_memory_info["total"].rmm_alloc_info["nbytes"] +
            alloc_info["nbytes"],
    })

    test_memory_info["total"].rmm_alloc_info["streams"].update({
        0:
            test_memory_info["total"].rmm_alloc_info["streams"][0] +
            alloc_info["streams"][0],
    })

    # Reset creating cupy arrays
    cp.cuda.set_allocator(None)


def pytest_terminal_summary(
        terminalreporter: _pytest.terminal.TerminalReporter,
        exitstatus: pytest.ExitCode,
        config: _pytest.config.Config):
    def hligh_py(obj):
        return highlight(str(obj), PythonLexer(),
                         TerminalFormatter(bg="dark")).rstrip("\n")

    terminalreporter.write_sep("=", "CumlArray Summary")

    terminalreporter.write_line("To Output Counts:", cyan=True)
    terminalreporter.write_line(hligh_py(_total_array_alloc.to_output_counts))

    terminalreporter.write_line("From Array Counts:", cyan=True)
    terminalreporter.write_line(hligh_py(_total_array_alloc.from_array_counts))

    terminalreporter.write_line("CuPy Malloc: Count={:,} Size={:,}".format(
        _total_array_alloc.cupy_alloc_info["count"],
        _total_array_alloc.cupy_alloc_info["nbytes"]))

    have_outstanding = list(
        filter(
            lambda x: "outstanding" in x[1].rmm_alloc_info and x[1].
            rmm_alloc_info["outstanding"] > 0,
            test_memory_info.items()))

    if (len(have_outstanding) > 0):
        terminalreporter.write_line("Memory leak in the following tests:",
                                    red=True)

        for key, memory in have_outstanding:
            terminalreporter.write_line(key)

    terminalreporter.write_line("Allocation Info: (test, peak, count)",
                                yellow=True)

    count = 0

    for key, memory in sorted(test_memory_info.items(),
                              key=lambda x: -x[1].rmm_alloc_info["peak"]):

        default_stream_nbytes = (memory.rmm_alloc_info["streams"][0] /
                                 memory.rmm_alloc_info["nbytes"] if
                                 memory.rmm_alloc_info["nbytes"] > 0 else 1.0)

        terminalreporter.ensure_newline()

        terminalreporter.write(("Peak={:>12,}, NBytes={:>12,}, Count={:>6,}"
                                ", Stream0={:>6.1%}, Test={}").format(
                                    memory.rmm_alloc_info["peak"],
                                    memory.rmm_alloc_info["nbytes"],
                                    memory.rmm_alloc_info["count"],
                                    default_stream_nbytes,
                                    key))

        if (memory.to_output_counts):
            terminalreporter.write(", ")
            terminalreporter.write("ToOutput: ", blue=True)
            terminalreporter.write(hligh_py(memory.to_output_counts))

        if (memory.from_array_counts):
            terminalreporter.write(", ")
            terminalreporter.write("FromArray: ", purple=True)
            terminalreporter.write(hligh_py(memory.from_array_counts))

        terminalreporter.line("")

        # if (count > 50):
        #     break

        count += 1
