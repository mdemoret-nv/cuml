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

import numbers
import os
import sys
import typing
import _pytest.config
import _pytest.terminal
import _pytest.python
import cupy as cp
import cupyx
import pytest
import rmm
import rmm._lib
from pytest import Item
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

# rmm.reinitialize(logging=True, log_file_name="test_log.txt")

callback_mr = rmm.mr.CallbackResourceAdaptor(
    rmm.mr.get_current_device_resource(), None)

rmm.mr.set_current_device_resource(callback_mr)

# Stores incorrect uses of CumlArray on cuml.common.base.Base to print at the
# end
bad_cuml_array_loc = set()

test_memory_info = {
    "total": {
        "count": 0, "peak": 0, "nbytes": 0, "streams": {
            0: 0,
        }
    }
}


def checked_isinstance(obj, class_name_dot_separated):
    """
    Small helper function to check instance of object that doesn't import
    class_path at import time, only at check time. Returns False if
    class_path cannot be imported.

    Parameters:
    -----------
    obj: Python object
        object to check if it is instance of a class
    class_name_dot_separated: list of str
        List of classes to check whether object is an instance of, each item
        can be a full dot  separated class like
        'cuml.dask.preprocessing.LabelEncoder'
    """
    ret = False
    for class_path in class_name_dot_separated:
        module_name, class_name = class_path.rsplit(".", 1)
        module = sys.modules[module_name]
        module_class = getattr(module, class_name, None)

        if module_class is not None:
            ret = isinstance(obj, module_class) or ret

    return ret


# Set a bad cupy allocator that will fail if rmm.rmm_cupy_allocator is not used
def bad_allocator(nbytes):

    assert False, \
        "Using default cupy allocator instead of rmm.rmm_cupy_allocator"

    return None


saved_allocator = rmm.rmm_cupy_allocator


def counting_rmm_allocator(nbytes):

    import cuml.common.array

    cuml.common.array._increment_malloc(nbytes)

    return saved_allocator(nbytes)


rmm.rmm_cupy_allocator = counting_rmm_allocator


def pytest_configure(config):
    cp.cuda.set_allocator(counting_rmm_allocator)


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
        # print("[MR] is_alloc: {}, mem_ptr: {}, n_nbytes: {}, stream_ptr: {}".format(is_alloc, mem_ptr, n_bytes, stream_ptr))
        try:
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
        except Exception as ex:
            print("Exception:")
            print(ex)


    callback_mr.set_callback(print_mr)

    # callback_mr = rmm.mr.CallbackResourceAdaptor(current_mr, print_mr)

    # rmm.mr.set_current_device_resource(callback_mr)

    yield

    import gc

    gc.collect()

    # assert len(allocations) == 0

    # del memory["outstanding"]

    test_memory_info[request.node.nodeid] = memory

    test_memory_info["total"].update({
        "count": test_memory_info["total"]["count"] + memory["count"],
        "peak": max(test_memory_info["total"]["peak"], memory["peak"]),
        "nbytes": test_memory_info["total"]["nbytes"] + memory["nbytes"],
    })

    test_memory_info["total"]["streams"].update({
        0: test_memory_info["total"]["streams"][0] + memory["streams"][0],
    })

    # Reset creating cupy arrays
    cp.cuda.set_allocator(None)


# @pytest.fixture(scope="module", autouse=True)
# def cuml_memory_per_module_fixture(request):

#     # Disable creating cupy arrays
#     # cp.cuda.set_allocator(bad_allocator)
#     cp.cuda.set_allocator(counting_rmm_allocator)

#     yield

#     # Reset creating cupy arrays
#     cp.cuda.set_allocator(None)


# Use the runtest_makereport hook to get the result of the test. This is
# necessary because pytest has some magic to extract the Cython source file
# from the traceback
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: Item, call):

    # Yield to the default implementation and get the result
    outcome = yield
    report = outcome.get_result()

    if (report.failed):

        # Save the abs path to this file. We will only mark bad CumlArray uses
        # if the assertion failure comes from this file
        conf_test_path = os.path.abspath(__file__)

        found_assert = False

        # Ensure these attributes exist. They can be missing if something else
        # failed outside of the test
        if (hasattr(report.longrepr, "reprtraceback")
                and hasattr(report.longrepr.reprtraceback, "reprentries")):

            for entry in reversed(report.longrepr.reprtraceback.reprentries):

                if (not found_assert and
                        entry.reprfileloc.message.startswith("AssertionError")
                        and os.path.abspath(
                            entry.reprfileloc.path) == conf_test_path):
                    found_assert = True
                elif (found_assert):
                    true_path = "{}:{}".format(entry.reprfileloc.path,
                                               entry.reprfileloc.lineno)

                    bad_cuml_array_loc.add(
                        (true_path, entry.reprfileloc.message))

                    break


def pytest_terminal_summary(
        terminalreporter: _pytest.terminal.TerminalReporter,
        exitstatus: pytest.ExitCode,
        config: _pytest.config.Config):

    terminalreporter.write_sep("=", "CumlArray Summary")

    import cuml.common.array

    terminalreporter.write_line("To Output Counts:", yellow=True)
    terminalreporter.write_line(str(cuml.common.array._to_output_counts))

    terminalreporter.write_line("From Array Counts:", yellow=True)
    terminalreporter.write_line(str(cuml.common.array._from_array_counts))

    terminalreporter.write_line("CuPy Malloc: Count={}, Size={}".format(
        cuml.common.array._malloc_count.get(),
        cuml.common.array._malloc_nbytes.get()))

    have_outstanding = list(
        filter(lambda x: "outstanding" in x[1] and x[1]["outstanding"] > 0,
               test_memory_info.items()))

    if (len(have_outstanding) > 0):
        terminalreporter.write_line("Memory leak in the following tests:",
                                    red=True)

        for key, memory in have_outstanding:
            terminalreporter.write_line(key)

    terminalreporter.write_line("Allocation Info: (test, peak, count)",
                                yellow=True)

    count = 0

    for key, memory in sorted(test_memory_info.items(), key=lambda x: -x[1]["peak"]):

        default_stream_nbytes = (memory["streams"][0] / memory["nbytes"]
                                 if memory["nbytes"] > 0 else 1.0)

        terminalreporter.write_line(
            "Peak={:>12,}, NBytes={:>12,}, Count={:>6,}, Stream0={:>6.1%}, Test={}"
            .format(memory["peak"],
                    memory["nbytes"],
                    memory["count"],
                    default_stream_nbytes,
                    key))

        # if (count > 50):
        #     break

        count += 1


# Closing hook to display the file/line numbers at the end of the test
def pytest_unconfigure(config):
    def split_exists(filename: str) -> bool:
        strip_colon = filename[:filename.rfind(":")]
        return os.path.exists(strip_colon)

    if (len(bad_cuml_array_loc) > 0):

        print("Incorrect CumlArray uses in class derived from "
              "cuml.common.base.Base:")

        prefix = ""

        # Depending on where pytest was launched from, it may need to append
        # "python"
        if (not os.path.basename(os.path.abspath(
                os.curdir)).endswith("python")):
            prefix = "python"

        for location, message in bad_cuml_array_loc:

            combined_path = os.path.abspath(location)

            # Try appending prefix if that file doesnt exist
            if (not split_exists(combined_path)):
                combined_path = os.path.abspath(os.path.join(prefix, location))

                # If that still doesnt exist, just use the original
                if (not split_exists(combined_path)):
                    combined_path = location

            print("{} {}".format(combined_path, message))

        print(
            "See https://github.com/rapidsai/cuml/issues/2456#issuecomment-666106406"  # noqa
            " for more information on naming conventions")


# This fixture will monkeypatch cuml.common.base.Base to check for incorrect
# uses of CumlArray.
@pytest.fixture(autouse=True)
def fail_on_bad_cuml_array_name(monkeypatch, request):

    if 'no_bad_cuml_array_check' in request.keywords:
        return

    from cuml.common import CumlArray
    from cuml.common.base import Base
    from cuml.common.input_utils import get_supported_input_type

    def patched__setattr__(self, name, value):

        if name == 'classes_' and \
                checked_isinstance(self,
                                   ['cuml.dask.preprocessing.LabelEncoder',
                                    'cuml.preprocessing.LabelEncoder']):
            # For label encoder, classes_ stores the set of unique classes
            # which is strings, and can't be saved as cuml array
            # even called `get_supported_input_type` causes a failure.
            pass
        else:
            supported_type = get_supported_input_type(value)

            if name == 'idf_':
                # We skip this test because idf_' for tfidf setter returns
                # a sparse diagonal matrix and getter gets a cupy array
                # see discussion at:
                # https://github.com/rapidsai/cuml/pull/2698/files#r471865982
                pass
            elif (supported_type == CumlArray):
                assert name.startswith("_"), \
                    ("Invalid CumlArray Use! CumlArray attributes need a "
                     "leading underscore. Attribute: '{}' "
                     "In: {}").format(name, self.__repr__())
            elif (supported_type == cp.ndarray
                  and cupyx.scipy.sparse.issparse(value)):
                # Leave sparse matrices alone for now.
                pass
            elif (supported_type is not None):
                if not isinstance(value, numbers.Number):
                    # Is this an estimated property?
                    # If so, should always be CumlArray
                    assert not name.endswith("_"), \
                        ("Invalid Estimated "
                         "Array-Like Attribute! Estimated attributes should "
                         "always be CumlArray. Attribute: '{}'"
                         " In: {}").format(name, self.__repr__())
                    assert not name.startswith("_"), \
                        ("Invalid Public "
                         "Array-Like Attribute! Public array-like attributes "
                         "should always be CumlArray. "
                         "Attribute: '{}' In: {}").format(name,
                                                          self.__repr__())
                else:
                    # Estimated properties can be numbers
                    pass

        return super(Base, self).__setattr__(name, value)

    # Monkeypatch CumlArray.__setattr__ to test for incorrect uses of
    # array-like objects
    # monkeypatch.setattr(Base, "__setattr__", patched__setattr__)


@pytest.fixture(scope="module")
def nlp_20news():
    twenty_train = fetch_20newsgroups(subset='train',
                                      shuffle=True,
                                      random_state=42)

    count_vect = CountVectorizer()
    X = count_vect.fit_transform(twenty_train.data)
    Y = cp.array(twenty_train.target)

    return X, Y


def pytest_addoption(parser):
    parser.addoption("--run_stress",
                     action="store_true",
                     default=False,
                     help="run stress tests")

    parser.addoption("--run_quality",
                     action="store_true",
                     default=False,
                     help="run quality tests")

    parser.addoption("--run_unit",
                     action="store_true",
                     default=False,
                     help="run unit tests")

    parser.addoption("--quick_run",
                     default=None,
                     type=int,
                     help="run unit tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run_quality"):
        # --run_quality given in cli: do not skip quality tests
        skip_stress = pytest.mark.skip(
            reason="Stress tests run with --run_stress flag.")
        for item in items:
            if "stress" in item.keywords:
                item.add_marker(skip_stress)
        skip_unit = pytest.mark.skip(
            reason="Stress tests run with --run_unit flag.")
        for item in items:
            if "unit" in item.keywords:
                item.add_marker(skip_unit)

    else:
        skip_quality = pytest.mark.skip(
            reason="Quality tests run with --run_quality flag.")
        for item in items:
            if "quality" in item.keywords:
                item.add_marker(skip_quality)

    if config.getoption("--run_stress"):
        # --run_stress given in cli: do not skip stress tests

        skip_unit = pytest.mark.skip(
            reason="Stress tests run with --run_unit flag.")
        for item in items:
            if "unit" in item.keywords:
                item.add_marker(skip_unit)

        skip_quality = pytest.mark.skip(
            reason="Quality tests run with --run_quality flag.")
        for item in items:
            if "quality" in item.keywords:
                item.add_marker(skip_quality)

    else:
        skip_stress = pytest.mark.skip(
            reason="Stress tests run with --run_stress flag.")
        for item in items:
            if "stress" in item.keywords:
                item.add_marker(skip_stress)

    quick_run = config.getoption("--quick_run")

    if (quick_run):
        root_node = {}
        leafs = []

        def get_leaf(node_list: list) -> list:

            curr_node = root_node

            for n in node_list:
                name = getattr(n, "originalname", n.name)

                if (name not in curr_node):
                    if (isinstance(n, _pytest.python.Function)):
                        curr_node[name] = []
                        leafs.append(curr_node[name])
                    else:
                        curr_node[name] = {}

                curr_node = curr_node[name]
            
            return curr_node

        for item in items:
            leaf = get_leaf(item.listchain())

            leaf.append(item)

        selected_items = []
        deselected_items = []

        def process_leaf_seeonce(leaf: typing.List[_pytest.python.Function]):
            seen = {}

            def has_been_seen(cs: _pytest.python.CallSpec2):
                for key, val in enumerate(cs._idlist):
                    if (key not in seen):
                        return False

                    if (val not in seen[key]):
                        return False

                return True

            def update_seen(cs: _pytest.python.CallSpec2):
                for key, val in enumerate(cs._idlist):
                    if (key not in seen):
                        seen[key] = []

                    if (val not in seen[key]):
                        seen[key].append(val)

            for l in leaf:
                # If no callspec, this is the only function call
                if (not hasattr(l, "callspec")):
                    selected_items.append(l)
                    continue

                callspec = l.callspec

                if (has_been_seen(callspec)):
                    deselected_items.append(l)
                else:
                    # Add to seen and selected
                    selected_items.append(l)

                    update_seen(callspec)


        for leaf in leafs:
            process_leaf_seeonce(leaf)

        config.hook.pytest_deselected(items=deselected_items)
        items[:] = selected_items

