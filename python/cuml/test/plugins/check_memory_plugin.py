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
import regex

import _pytest.config
import _pytest.python
import _pytest.terminal
import _pytest.nodes
import cupy as cp
import pytest
import rmm
import rmm._lib
from pygments import highlight
from pygments.formatters.terminal import TerminalFormatter
from pygments.lexers.python import PythonLexer

_F = typing.TypeVar("_F", bound=typing.Callable[..., typing.Any])


def _get_empty_alloc_info():
    return {
        "peak": 0,
        "count": 0,
        "nbytes": 0,
        "outstanding": 0,
    }


def _add_dict(dict1: dict, dict2: dict) -> dict:

    output = {}

    for key in set(dict1) | set(dict2):
        val1 = dict1.get(key, 0)
        val2 = dict2.get(key, 0)

        if (isinstance(val1, dict) or isinstance(val2, dict)):
            output[key] = _add_dict(dict1.get(key, {}), dict2.get(key, {}))
        elif key == "peak":
            output[key] = max(val1, val2)
        else:
            output[key] = val1 + val2

    # return {
    #     key: _add_dict(dict1.get(key, {}), dict2.get(key, {})) if
    #     (isinstance(dict1.get(key, 0), dict)
    #      or isinstance(dict2.get(key, 0), dict)) else dict1.get(key, 0) +
    #     dict2.get(key, 0)
    #     for key in set(dict1) | set(dict2)
    # }
    return output


class MemoryNode:
    def __init__(self, parent: "MemoryNode", name: str) -> None:
        self._parent = parent
        self._name = name

        self._children: typing.Dict[str, MemoryNode] = {}

        self._to_output_counts = {}
        self._from_array_counts = {}
        self._rmm_alloc_info = _get_empty_alloc_info()
        self._cupy_alloc_info = _get_empty_alloc_info()

    @property
    def name(self):
        return self._parent.name + self._name

    @property
    def to_output_counts(self):
        total = self._to_output_counts

        for child in self._children.values():

            total = _add_dict(total, child.to_output_counts)

        return total

    @property
    def from_array_counts(self):
        total = self._from_array_counts

        for child in self._children.values():

            total = _add_dict(total, child.from_array_counts)

        return total

    @property
    def rmm_alloc_info(self):
        total = self._rmm_alloc_info

        for child in self._children.values():

            total = _add_dict(total, child.rmm_alloc_info)

        return total

    @property
    def cupy_alloc_info(self):
        total = self._cupy_alloc_info

        for child in self._children.values():

            total = _add_dict(total, child.cupy_alloc_info)

        return total

    def get_child(self, full_name: str) -> "MemoryNode":

        front_idx = full_name.index(self.name) + len(self.name)

        if (front_idx == -1):
            # If we dont start with our own name, then it MUST match exactly in
            # the children list
            assert full_name in self._children

            return self._children[full_name]

        node_match = regex.search(r'/(?=.)|::(?=.)|.(?=\[.*\])',
                                  full_name,
                                  pos=front_idx)

        if (node_match):
            remaining_name = full_name[:node_match.end()]
            # Get the child
            child = self.get_child(remaining_name)

            return child.get_child(full_name)
        else:
            # Need to find our direct child
            child_name = full_name[front_idx:]

            if (child_name not in self._children):
                self._children[child_name] = MemoryNode(parent=self,
                                                        name=child_name)

            return self._children[child_name]

        # # Now see if we need to make any intermediate nodes
        # while (front_idx < len(full_name)):
        #     next_idx = full_name.index(self._name)

        # # First, split off our own name if it is there
        # if (name_segments.startswith(self._name)):
        #     name_segments = name_segments[:len(self._name)]

        # if (isinstance(name_segments, str)):
        #     if (name_segments not in self._children):
        #         self._children[name_segments] = MemoryNode(parent=self,
        #                                                    name=name_segments)

        #     return self._children[name_segments]

        # assert (isinstance(name_segments, list))

        # # Must be a tuple. Pop the first item and recursively call
        # first_segment = name_segments[0]

        # inner_child = self.get_child(first_segment)

        # if (len(name_segments) > 1):
        #     return inner_child.get_child(name_segments[1:])

        # return inner_child

    def increment_to_output(self, output_type: str):
        self._to_output_counts[
            output_type] = self._to_output_counts.setdefault(output_type,
                                                             0) + 1

    def increment_from_output(self, output_type: str):
        self._from_array_counts[
            output_type] = self._from_array_counts.setdefault(output_type,
                                                              0) + 1

    def increment_cupy_alloc(self, nbytes: int):
        self._cupy_alloc_info["count"] += 1
        self._cupy_alloc_info["nbytes"] += nbytes

    def set_rmm_alloc_info(self, values: dict):
        self._rmm_alloc_info = values

    def set_cupy_alloc_info(self, values: dict):
        self._cupy_alloc_info = values


class RootMemoryNode(MemoryNode):
    def __init__(self) -> None:
        super().__init__(name="Root", parent=None)

    @property
    def name(self):
        return ""


def pytest_addoption(parser):

    group = parser.getgroup('cuML Memory Checker')

    group.addoption(
        "--check_memory",
        action="store_true",
        default=False,
        help=("Adds a memory checker plugin that reports tests with memory "
              "leaks"))


def pytest_configure(config: _pytest.config.Config):

    is_enabled = any(config.getvalue(x) for x in ('check_memory', ))

    # Exit early if we arent enabled
    if not is_enabled:
        return

    plugin_instance = CheckMemoryPlugin()

    config.pluginmanager.register(plugin_instance, "_check_memory_plugin")


class CheckMemoryPlugin(object):
    def __init__(self):

        pool_mr = rmm.mr.PoolMemoryResource(rmm.mr.get_current_device_resource())

        # Create a global tracking MR to watch all allocations
        self.global_tracked_mr = rmm.mr.TrackingMemoryResource(pool_mr)

        rmm.mr.set_current_device_resource(self.global_tracked_mr)

        # Create the tracking classes
        self.test_memory_info: typing.Dict[str, MemoryNode] = {
            "root": RootMemoryNode()
        }

        self._total_memory_alloc = RootMemoryNode()
        self._test_memory_alloc: MemoryNode = None

        # Monkeypatch the rmm cupy allocator to track allocations
        saved_allocator = rmm.rmm_cupy_allocator

        def counting_rmm_allocator(nbytes):

            self._increment_malloc(nbytes)

            return saved_allocator(nbytes)

        rmm.rmm_cupy_allocator = counting_rmm_allocator

        cp.cuda.set_allocator(counting_rmm_allocator)

        # Finally, Monkeypatch the CumlArray.to_output and input_to_cuml_array
        import cuml.common.array
        import cuml.common.input_utils

        saved_self = self

        def tracked_to_output(func: _F) -> _F:
            @wraps(func)
            def inner(self, output_type='cupy', output_dtype=None):

                saved_self._increment_to_output(output_type)

                return func(self,
                            output_type=output_type,
                            output_dtype=output_dtype)

            return inner

        def tracked_input_to_cuml_array(func: _F) -> _F:
            @wraps(func)
            def inner(X, *args, **kwargs):

                saved_self._increment_from_array(
                    cuml.common.input_utils.determine_array_type(X))

                return func(X, *args, **kwargs)

            return inner

        cuml.common.array.CumlArray.to_output = tracked_to_output(
            cuml.common.array.CumlArray.to_output)
        cuml.common.input_utils.input_to_cuml_array = \
            tracked_input_to_cuml_array(
                cuml.common.input_utils.input_to_cuml_array)

    def _increment_to_output(self, output_type: str):

        if (self._test_memory_alloc):
            self._test_memory_alloc.increment_to_output(output_type)

        self._total_memory_alloc.increment_to_output(output_type)

    def _increment_from_array(self, output_type: str):

        if (self._test_memory_alloc):
            self._test_memory_alloc.increment_from_output(output_type)

        self._total_memory_alloc.increment_from_output(output_type)

    def _increment_malloc(self, nbytes: int):

        if (self._test_memory_alloc):
            self._test_memory_alloc.increment_cupy_alloc(nbytes=nbytes)

        self._total_memory_alloc.increment_cupy_alloc(nbytes=nbytes)

    @pytest.hookimpl(hookwrapper=True, trylast=True)
    def pytest_runtest_call(self, item: _pytest.nodes.Item):
        # @pytest.fixture(scope="function", autouse=True)
        # def cupy_allocator_fixture(self, request):

        # Set a bad cupy allocator that will fail if rmm.rmm_cupy_allocator is
        # not used
        def bad_allocator(nbytes):

            assert False, \
                "Using default cupy allocator. Use rmm.rmm_cupy_allocator"

        # Disable creating cupy arrays
        # cp.cuda.set_allocator(bad_allocator)
        cp.cuda.set_allocator(rmm.rmm_cupy_allocator)

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
                memory[
                    "outstanding"] -= n_bytes if n_bytes > 0 else popped_nbytes

        # callback_mr.set_callback(print_mr)

        # global_alloc_counts_pre = global_tracked_mr.get_allocation_counts()

        # old_mr = rmm.mr.get_current_device_resource()

        try:
            # tracked_mr = rmm.mr.TrackingMemoryResource(old_mr, True)

            # rmm.mr.set_current_device_resource(tracked_mr)

            self.global_tracked_mr.push_allocation_counts()

            self._test_memory_alloc = self.test_memory_info["root"].get_child(
                item.nodeid)

            # import cuml
            import cupy

            # cuml.cuda.nvtx_range_push(item.nodeid)
            cupy.cuda.nvtx.RangePush(item.nodeid)

            yield

            # cuml.cuda.nvtx_range_pop()
            cupy.cuda.nvtx.RangePop()

            import gc

            gc.collect()

            temp_alloc_info = self.global_tracked_mr.pop_allocation_counts()

            alloc_info = {
                "peak": temp_alloc_info["peak_bytes"],
                "count": temp_alloc_info["total_count"],
                "nbytes": temp_alloc_info["total_bytes"],
                "outstanding": temp_alloc_info["current_bytes"],
            }

            # if (alloc_info["outstanding"] > 0):
            #     print(self.global_tracked_mr.get_outstanding_allocations_str())

            self._test_memory_alloc.set_rmm_alloc_info(alloc_info)

            self.test_memory_info[item.nodeid] = self._test_memory_alloc

        finally:
            # rmm.mr.set_current_device_resource(old_mr)
            self._test_memory_alloc = None

            # Reset creating cupy arrays
            cp.cuda.set_allocator(None)

    def pytest_terminal_summary(
            self, terminalreporter: _pytest.terminal.TerminalReporter):
        def hligh_py(obj):
            return highlight(str(obj),
                             PythonLexer(),
                             TerminalFormatter(bg="dark")).rstrip("\n")

        root_node = self.test_memory_info["root"]

        terminalreporter.write_sep("=", "CumlArray Summary")

        terminalreporter.write_line("To Output Counts:", cyan=True)
        terminalreporter.write_line(hligh_py(root_node.to_output_counts))

        terminalreporter.write_line("From Array Counts:", cyan=True)
        terminalreporter.write_line(hligh_py(root_node.from_array_counts))

        terminalreporter.write_line("CuPy Malloc: Count={:,} Size={:,}".format(
            root_node.cupy_alloc_info["count"],
            root_node.cupy_alloc_info["nbytes"]))

        have_outstanding = list(
            filter(
                lambda x: "outstanding" in x[1].rmm_alloc_info and x[1].
                rmm_alloc_info["outstanding"] > 0,
                self.test_memory_info.items()))

        if (len(have_outstanding) > 0):
            terminalreporter.write_line("Memory leak in the following tests:",
                                        red=True)

            for key, memory in have_outstanding:
                terminalreporter.write_line("{} - {} B".format(
                    key, memory.rmm_alloc_info["outstanding"]))

        terminalreporter.write_line("Allocation Info: (test, peak, count)",
                                    yellow=True)

        count = 0

        root_state = {"indent": 0, "empty_cols": [], "is_last": True}

        markup = terminalreporter._tw.markup

        def _print_line(string: str,
                        indent: int,
                        empty_cols: list,
                        is_last: bool):
            terminalreporter.ensure_newline()

            indent_str = ""

            # Start by printing any indentation
            for i in range(indent):

                indent_str += "│" if i not in empty_cols else " "
                indent_str += " " * 2

            # Now we need the last indent for our row
            indent_str += ('└─ ' if is_last else '├─ ')

            terminalreporter.write(indent_str)

            # Now print name
            terminalreporter.write(string)

            terminalreporter.line("")

        def print_node(node_memory: MemoryNode, state: dict):

            indent = state["indent"]
            empty_cols = state["empty_cols"]
            is_last = state["is_last"]

            has_children = len(node_memory._children) > 0

            to_print = markup(node_memory._name, bold=True, cyan=True)

            alloc_info = node_memory.rmm_alloc_info

            if (has_children):
                to_print += markup(" = Peak={:,}".format(alloc_info["peak"]))
                to_print += markup(", NBytes={:,}, Count={:,}".format(
                    alloc_info["nbytes"], alloc_info["count"]),
                                   light=True)

            _print_line(to_print, indent, empty_cols, is_last)

            # Now, if we have children, process those
            if (has_children):
                state["indent"] += 1

                sorted_children = sorted(
                    node_memory._children.values(),
                    key=lambda x: -x.rmm_alloc_info["peak"])

                for idx, child in enumerate(sorted_children):

                    if state["is_last"]:
                        state["empty_cols"].append(indent)

                    state["is_last"] = True if idx == len(
                        node_memory._children) - 1 else False

                    state = print_node(child, state)
            else:

                _print_line(markup(str(alloc_info), light=True),
                            indent + 1,
                            empty_cols + [indent] if is_last else empty_cols,
                            True)

            if (is_last and indent != 0):

                state["indent"] -= 1

                if indent in empty_cols:
                    state["empty_cols"].remove(indent)

                state["is_last"] = False

            return state

        print_node(root_node, root_state)

        max_peak = 50

        terminalreporter.write_sep(
            "_", "Largest Peak Allocations ({})".format(max_peak))
        for key, memory in sorted(self.test_memory_info.items(),
                                  key=lambda x: -x[1].rmm_alloc_info["peak"]):

            # default_stream_nbytes = (memory.rmm_alloc_info["streams"][0] /
            #                          memory.rmm_alloc_info["nbytes"]
            #                          if memory.rmm_alloc_info["nbytes"] > 0
            #                          else 1.0)

            terminalreporter.ensure_newline()

            terminalreporter.write(
                ("Peak={:>12,}, NBytes={:>14,}, Count={:>8,}"
                 ", Test={}").format(memory.rmm_alloc_info["peak"],
                                     memory.rmm_alloc_info["nbytes"],
                                     memory.rmm_alloc_info["count"],
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

            if (count > max_peak):
                break

            count += 1
