# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for APIC (API Capture) - CUDA Graph Serialization and Replay."""

import os
import tempfile
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


# Simple kernel for testing
@wp.kernel
def scale_kernel(input: wp.array(dtype=float), output: wp.array(dtype=float), scale: float):
    tid = wp.tid()
    output[tid] = input[tid] * scale


@wp.kernel
def add_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]


@wp.kernel
def saxpy_kernel(x: wp.array(dtype=float), y: wp.array(dtype=float), a: float):
    """y = a * x + y"""
    tid = wp.tid()
    y[tid] = a * x[tid] + y[tid]


def test_apic_capture_begin_end(test, device):
    """Test basic APIC capture begin/end."""
    n = 1024

    input_data = wp.array(np.ones(n, dtype=np.float32), device=device)
    output_data = wp.zeros(n, dtype=float, device=device)

    # Capture with APIC enabled
    wp.capture_begin(device=device, apic=True)
    wp.launch(scale_kernel, dim=n, inputs=[input_data, output_data, 2.0], device=device)
    graph = wp.capture_end(device=device)

    # Verify APIC capture was created
    test.assertIsNotNone(graph.apic_capture)
    test.assertEqual(len(graph.apic_capture.launches), 1)
    test.assertEqual(len(graph.apic_capture.modules), 1)
    test.assertEqual(len(graph.apic_capture.kernels), 1)


def test_apic_scoped_capture(test, device):
    """Test APIC with ScopedCapture context manager."""
    n = 1024

    input_data = wp.array(np.ones(n, dtype=np.float32), device=device)
    output_data = wp.zeros(n, dtype=float, device=device)

    with wp.ScopedCapture(device=device, apic=True) as capture:
        wp.launch(scale_kernel, dim=n, inputs=[input_data, output_data, 3.0], device=device)

    # Verify APIC capture
    test.assertIsNotNone(capture.graph.apic_capture)
    test.assertEqual(len(capture.graph.apic_capture.launches), 1)

    # Execute the graph normally to verify it still works
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    # Check result
    result = output_data.numpy()
    expected = np.ones(n, dtype=np.float32) * 3.0
    np.testing.assert_array_almost_equal(result, expected)


def test_apic_multiple_launches(test, device):
    """Test APIC with multiple kernel launches."""
    n = 1024

    a = wp.array(np.ones(n, dtype=np.float32), device=device)
    b = wp.array(np.ones(n, dtype=np.float32) * 2, device=device)
    c = wp.zeros(n, dtype=float, device=device)
    d = wp.zeros(n, dtype=float, device=device)

    with wp.ScopedCapture(device=device, apic=True) as capture:
        # First kernel: c = a + b
        wp.launch(add_kernel, dim=n, inputs=[a, b, c], device=device)
        # Second kernel: d = c * 2
        wp.launch(scale_kernel, dim=n, inputs=[c, d, 2.0], device=device)

    # Verify multiple launches were recorded
    test.assertEqual(len(capture.graph.apic_capture.launches), 2)

    # Execute and verify
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    # c should be 1 + 2 = 3
    # d should be 3 * 2 = 6
    np.testing.assert_array_almost_equal(c.numpy(), np.ones(n) * 3)
    np.testing.assert_array_almost_equal(d.numpy(), np.ones(n) * 6)


def test_apic_memory_regions(test, device):
    """Test APIC memory region tracking."""
    n = 1024

    input_data = wp.array(np.arange(n, dtype=np.float32), device=device)
    output_data = wp.zeros(n, dtype=float, device=device)

    with wp.ScopedCapture(device=device, apic=True) as capture:
        wp.launch(scale_kernel, dim=n, inputs=[input_data, output_data, 2.0], device=device)

    apic = capture.graph.apic_capture

    # Should have tracked memory regions for input and output arrays
    test.assertGreaterEqual(len(apic.memory_regions), 2)


def test_apic_save_load_basic(test, device):
    """Test basic save and load functionality."""
    n = 256

    input_data = wp.array(np.ones(n, dtype=np.float32) * 5.0, device=device)
    output_data = wp.zeros(n, dtype=float, device=device)

    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = os.path.join(tmpdir, "test_graph")

        # Capture with APIC
        with wp.ScopedCapture(device=device, apic=True) as capture:
            wp.launch(scale_kernel, dim=n, inputs=[input_data, output_data, 2.0], device=device)

        # Save the graph
        wp.capture_save(
            capture.graph,
            graph_path,
            inputs={"input": input_data},
            outputs={"output": output_data},
        )

        # Verify files were created
        test.assertTrue(os.path.exists(graph_path + ".wgf"))
        test.assertTrue(os.path.exists(graph_path + "_modules"))


def test_apic_wgf_format(test, device):
    """Test WGF file format reading and writing."""
    from warp._src.apic import WGFReader, WGFWriter

    with tempfile.TemporaryDirectory() as tmpdir:
        wgf_path = os.path.join(tmpdir, "test.wgf")

        # Write a test file
        writer = WGFWriter(wgf_path, device.arch)
        writer.add_metadata({"test_key": "test_value", "version": 1})
        writer.add_memory(b"test memory data")
        writer.add_operations(b"test operations data")
        writer.write()

        # Read it back
        reader = WGFReader(wgf_path)

        test.assertEqual(reader.target_arch, device.arch)

        metadata = reader.get_metadata()
        test.assertEqual(metadata["test_key"], "test_value")
        test.assertEqual(metadata["version"], 1)

        test.assertEqual(reader.get_memory(), b"test memory data")
        test.assertEqual(reader.get_operations(), b"test operations data")


def test_apic_capture_class(test, device):
    """Test APICapture class directly."""
    from warp._src.apic import APICapture, MemoryRole

    n = 512

    input_arr = wp.array(np.ones(n, dtype=np.float32), device=device)
    output_arr = wp.zeros(n, dtype=float, device=device)

    # Create capture and track arrays
    apic = APICapture(device)

    # Track arrays
    _region_id_in, offset_in = apic.track_array(input_arr, MemoryRole.INPUT)
    _region_id_out, offset_out = apic.track_array(output_arr, MemoryRole.OUTPUT)

    test.assertEqual(offset_in, 0)  # Base array, no offset
    test.assertEqual(offset_out, 0)  # Base array, no offset
    test.assertIn(input_arr.ptr, apic.memory_regions)
    test.assertIn(output_arr.ptr, apic.memory_regions)


def test_apic_array_slicing(test, device):
    """Test APIC handles array slicing/aliasing correctly."""
    from warp._src.apic import APICapture, MemoryRole

    n = 1024

    # Create a base array and slices
    base_arr = wp.array(np.arange(n, dtype=np.float32), device=device)
    slice1 = base_arr[0:512]
    slice2 = base_arr[512:1024]

    apic = APICapture(device)

    # Track slices - they should resolve to the same region with different offsets
    region_id_base, offset_base = apic.track_array(base_arr, MemoryRole.INTERNAL)
    region_id_1, offset_1 = apic.track_array(slice1, MemoryRole.INPUT)
    region_id_2, offset_2 = apic.track_array(slice2, MemoryRole.OUTPUT)

    # All should map to the same region
    test.assertEqual(region_id_base, region_id_1)
    test.assertEqual(region_id_base, region_id_2)

    # Offsets should be different
    test.assertEqual(offset_base, 0)
    test.assertEqual(offset_1, 0)  # slice1 starts at beginning
    test.assertEqual(offset_2, 512 * 4)  # slice2 starts at 512 * sizeof(float)


def test_apic_input_output_bindings(test, device):
    """Test input/output binding functionality."""
    from warp._src.apic import APICapture, MemoryRole

    n = 256

    input_arr = wp.array(np.ones(n, dtype=np.float32), device=device)
    output_arr = wp.zeros(n, dtype=float, device=device)

    apic = APICapture(device)

    # Set bindings
    apic.set_input_binding("my_input", input_arr)
    apic.set_output_binding("my_output", output_arr)

    # Verify bindings
    test.assertIn("my_input", apic.input_bindings)
    test.assertIn("my_output", apic.output_bindings)

    # Check roles were updated
    input_region = apic.memory_regions[input_arr.ptr]
    output_region = apic.memory_regions[output_arr.ptr]

    test.assertEqual(input_region.role, MemoryRole.INPUT)
    test.assertEqual(output_region.role, MemoryRole.OUTPUT)


def test_apic_kernel_info_tracking(test, device):
    """Test that kernel information is properly tracked."""
    n = 128

    a = wp.array(np.ones(n, dtype=np.float32), device=device)
    b = wp.array(np.ones(n, dtype=np.float32) * 2, device=device)
    c = wp.zeros(n, dtype=float, device=device)

    with wp.ScopedCapture(device=device, apic=True) as capture:
        wp.launch(add_kernel, dim=n, inputs=[a, b, c], device=device)

    apic = capture.graph.apic_capture

    # Check kernel info
    test.assertEqual(len(apic.kernels), 1)

    kernel_key = next(iter(apic.kernels.keys()))
    kernel_info = apic.kernels[kernel_key]

    test.assertIn("add_kernel", kernel_key)
    test.assertIsNotNone(kernel_info.forward_name)
    test.assertGreater(kernel_info.block_dim, 0)


def test_apic_launch_record(test, device):
    """Test launch record details."""
    n = 512

    x = wp.array(np.ones(n, dtype=np.float32), device=device)
    y = wp.array(np.ones(n, dtype=np.float32) * 2, device=device)

    with wp.ScopedCapture(device=device, apic=True) as capture:
        wp.launch(saxpy_kernel, dim=n, inputs=[x, y, 3.0], device=device)

    apic = capture.graph.apic_capture

    test.assertEqual(len(apic.launches), 1)

    launch = apic.launches[0]
    test.assertEqual(launch.dim, n)
    test.assertTrue(launch.is_forward)
    test.assertGreater(len(launch.param_bindings), 0)

    # Check parameter bindings
    array_bindings = [b for b in launch.param_bindings if b["type"] == "array"]
    scalar_bindings = [b for b in launch.param_bindings if b["type"] == "scalar"]

    test.assertEqual(len(array_bindings), 2)  # x and y
    test.assertEqual(len(scalar_bindings), 1)  # a (the scalar)


def test_apic_graph_execution_unchanged(test, device):
    """Test that normal graph execution still works with APIC enabled."""
    n = 1024

    input_data = wp.array(np.arange(n, dtype=np.float32), device=device)
    output_data = wp.zeros(n, dtype=float, device=device)

    # Capture with APIC
    with wp.ScopedCapture(device=device, apic=True) as capture:
        wp.launch(scale_kernel, dim=n, inputs=[input_data, output_data, 0.5], device=device)

    # Execute the graph multiple times
    for _ in range(3):
        wp.capture_launch(capture.graph)

    wp.synchronize_device(device)

    # Verify result (should be input * 0.5 after any number of replays since
    # we're reading from the same input each time)
    expected = np.arange(n, dtype=np.float32) * 0.5
    np.testing.assert_array_almost_equal(output_data.numpy(), expected)


def test_apic_save_load_execute(test, device):
    """Test full round-trip: capture, save, load, and execute."""
    n = 256

    # Create input data
    input_data = wp.array(np.arange(n, dtype=np.float32) + 1.0, device=device)  # [1, 2, 3, ...]
    output_data = wp.zeros(n, dtype=float, device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = os.path.join(tmpdir, "test_graph")

        # Capture with APIC
        with wp.ScopedCapture(device=device, apic=True) as capture:
            wp.launch(scale_kernel, dim=n, inputs=[input_data, output_data, 3.0], device=device)

        # Save the graph with bindings
        wp.capture_save(
            capture.graph,
            graph_path,
            inputs={"input": input_data},
            outputs={"output": output_data},
        )

        # Load the graph
        loaded_graph = wp.capture_load(graph_path, device=device)

        # Create new arrays for execution
        new_input = wp.array(np.ones(n, dtype=np.float32) * 10.0, device=device)
        new_output = wp.zeros(n, dtype=float, device=device)

        # Bind the new arrays
        loaded_graph.bind_input("input", new_input)
        loaded_graph.bind_output("output", new_output)

        # Execute using capture_launch (works with both captured and loaded graphs)
        wp.capture_launch(loaded_graph)
        wp.synchronize_device(device)

        # Verify result: 10.0 * 3.0 = 30.0
        expected = np.ones(n, dtype=np.float32) * 30.0
        np.testing.assert_array_almost_equal(new_output.numpy(), expected)


def test_apic_load_execute_multiple_kernels(test, device):
    """Test loading and executing a graph with multiple kernels."""
    n = 128

    a = wp.array(np.ones(n, dtype=np.float32) * 2.0, device=device)
    b = wp.array(np.ones(n, dtype=np.float32) * 3.0, device=device)
    c = wp.zeros(n, dtype=float, device=device)
    d = wp.zeros(n, dtype=float, device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = os.path.join(tmpdir, "multi_kernel_graph")

        # Capture multiple kernels
        with wp.ScopedCapture(device=device, apic=True) as capture:
            wp.launch(add_kernel, dim=n, inputs=[a, b, c], device=device)  # c = a + b = 5
            wp.launch(scale_kernel, dim=n, inputs=[c, d, 2.0], device=device)  # d = c * 2 = 10

        # Save
        wp.capture_save(
            capture.graph,
            graph_path,
            inputs={"a": a, "b": b},
            outputs={"c": c, "d": d},
        )

        # Load
        loaded_graph = wp.capture_load(graph_path, device=device)

        # Create new arrays
        new_a = wp.array(np.ones(n, dtype=np.float32) * 5.0, device=device)
        new_b = wp.array(np.ones(n, dtype=np.float32) * 7.0, device=device)
        new_c = wp.zeros(n, dtype=float, device=device)
        new_d = wp.zeros(n, dtype=float, device=device)

        # Bind
        loaded_graph.bind_input("a", new_a)
        loaded_graph.bind_input("b", new_b)
        loaded_graph.bind_output("c", new_c)
        loaded_graph.bind_output("d", new_d)

        # Execute using capture_launch (works with both captured and loaded graphs)
        wp.capture_launch(loaded_graph)
        wp.synchronize_device(device)

        # Verify: c = 5 + 7 = 12, d = 12 * 2 = 24
        np.testing.assert_array_almost_equal(new_c.numpy(), np.ones(n) * 12.0)
        np.testing.assert_array_almost_equal(new_d.numpy(), np.ones(n) * 24.0)


def test_apic_with_memory_ops(test, device):
    """Test APIC with memory operations (wp.copy) in addition to kernel launches."""
    n = 256

    # Create arrays
    src = wp.array(np.arange(n, dtype=np.float32) + 1.0, device=device)  # [1, 2, 3, ...]
    tmp = wp.zeros(n, dtype=float, device=device)
    dst = wp.zeros(n, dtype=float, device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = os.path.join(tmpdir, "memop_graph")

        # Capture a graph that:
        # 1. Copies src to tmp (D2D copy)
        # 2. Scales tmp and writes to dst (kernel)
        with wp.ScopedCapture(device=device, apic=True) as capture:
            wp.copy(tmp, src)
            wp.launch(scale_kernel, dim=n, inputs=[tmp, dst, 2.0], device=device)

        # Verify memory op was recorded
        apic = capture.graph.apic_capture
        test.assertEqual(len(apic.memory_ops), 1)
        test.assertEqual(len(apic.launches), 1)
        test.assertEqual(apic.memory_ops[0].kind, "D2D")

        # Execute original graph to verify it works
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)

        # dst should be (src * 2) = [2, 4, 6, ...]
        expected = (np.arange(n, dtype=np.float32) + 1.0) * 2.0
        np.testing.assert_array_almost_equal(dst.numpy(), expected)

        # Save the graph
        wp.capture_save(
            capture.graph,
            graph_path,
            inputs={"src": src},
            outputs={"dst": dst},
        )

        # Load the graph
        loaded_graph = wp.capture_load(graph_path, device=device)

        # Create new arrays for execution
        new_src = wp.array(np.ones(n, dtype=np.float32) * 10.0, device=device)
        new_dst = wp.zeros(n, dtype=float, device=device)

        # Bind the new arrays
        loaded_graph.bind_input("src", new_src)
        loaded_graph.bind_output("dst", new_dst)

        # Execute
        wp.capture_launch(loaded_graph)
        wp.synchronize_device(device)

        # new_dst should be (10.0 * 2.0) = 20.0
        expected = np.ones(n, dtype=np.float32) * 20.0
        np.testing.assert_array_almost_equal(new_dst.numpy(), expected)


def test_apic_complex_pipeline(test, device):
    """Test a more complex pipeline with multiple kernels and memory operations."""
    n = 128

    # Create arrays for a multi-stage pipeline:
    # Stage 1: a + b -> c (add_kernel)
    # Stage 2: copy c -> d
    # Stage 3: d * 2 -> e (scale_kernel)
    # Stage 4: copy e -> f
    # Stage 5: f + c -> g (add_kernel)
    a = wp.array(np.ones(n, dtype=np.float32) * 2.0, device=device)
    b = wp.array(np.ones(n, dtype=np.float32) * 3.0, device=device)
    c = wp.zeros(n, dtype=float, device=device)
    d = wp.zeros(n, dtype=float, device=device)
    e = wp.zeros(n, dtype=float, device=device)
    f = wp.zeros(n, dtype=float, device=device)
    g = wp.zeros(n, dtype=float, device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = os.path.join(tmpdir, "complex_pipeline")

        with wp.ScopedCapture(device=device, apic=True) as capture:
            wp.launch(add_kernel, dim=n, inputs=[a, b, c], device=device)  # c = 2 + 3 = 5
            wp.copy(d, c)  # d = 5
            wp.launch(scale_kernel, dim=n, inputs=[d, e, 2.0], device=device)  # e = 5 * 2 = 10
            wp.copy(f, e)  # f = 10
            wp.launch(add_kernel, dim=n, inputs=[f, c, g], device=device)  # g = 10 + 5 = 15

        apic = capture.graph.apic_capture
        test.assertEqual(len(apic.launches), 3)
        test.assertEqual(len(apic.memory_ops), 2)

        # Execute and verify
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)

        np.testing.assert_array_almost_equal(c.numpy(), np.ones(n) * 5.0)
        np.testing.assert_array_almost_equal(d.numpy(), np.ones(n) * 5.0)
        np.testing.assert_array_almost_equal(e.numpy(), np.ones(n) * 10.0)
        np.testing.assert_array_almost_equal(f.numpy(), np.ones(n) * 10.0)
        np.testing.assert_array_almost_equal(g.numpy(), np.ones(n) * 15.0)

        # Save
        wp.capture_save(
            capture.graph,
            graph_path,
            inputs={"a": a, "b": b},
            outputs={"g": g},
        )

        # Load
        loaded_graph = wp.capture_load(graph_path, device=device)

        # Create new input/output arrays
        new_a = wp.array(np.ones(n, dtype=np.float32) * 10.0, device=device)
        new_b = wp.array(np.ones(n, dtype=np.float32) * 5.0, device=device)
        new_g = wp.zeros(n, dtype=float, device=device)

        # Bind
        loaded_graph.bind_input("a", new_a)
        loaded_graph.bind_input("b", new_b)
        loaded_graph.bind_output("g", new_g)

        # Execute
        wp.capture_launch(loaded_graph)
        wp.synchronize_device(device)

        # Expected: c = 10 + 5 = 15, e = 15 * 2 = 30, g = 30 + 15 = 45
        expected_g = np.ones(n, dtype=np.float32) * 45.0
        np.testing.assert_array_almost_equal(new_g.numpy(), expected_g)


def test_apic_internal_allocation(test, device):
    """Test APIC with memory allocation inside graph capture."""
    n = 128

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = os.path.join(tmpdir, "alloc_graph")

        # Create input/output arrays BEFORE capture
        input_data = wp.array(np.arange(n, dtype=np.float32) + 1.0, device=device)
        output_data = wp.zeros(n, dtype=float, device=device)

        with wp.ScopedCapture(device=device, apic=True) as capture:
            # Allocate temporary array INSIDE the capture
            tmp = wp.zeros(n, dtype=float, device=device)
            # Pipeline: tmp = input * 2, output = tmp + input
            wp.launch(scale_kernel, dim=n, inputs=[input_data, tmp, 2.0], device=device)
            wp.launch(add_kernel, dim=n, inputs=[tmp, input_data, output_data], device=device)

        apic = capture.graph.apic_capture
        # Should have 3 memory regions: input, output, and internal tmp
        test.assertEqual(len(apic.memory_regions), 3)

        # Execute original graph to verify it works
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)

        # tmp = [2, 4, 6, ...], output = tmp + input = [3, 6, 9, ...]
        expected = (np.arange(n, dtype=np.float32) + 1.0) * 3.0
        np.testing.assert_array_almost_equal(output_data.numpy(), expected)

        # Save
        wp.capture_save(
            capture.graph,
            graph_path,
            inputs={"input": input_data},
            outputs={"output": output_data},
        )

        # Load
        loaded_graph = wp.capture_load(graph_path, device=device)

        # Should have allocated memory for all 3 regions
        test.assertEqual(len(loaded_graph._memory_regions), 3)

        # Create new arrays
        new_input = wp.array(np.ones(n, dtype=np.float32) * 10.0, device=device)
        new_output = wp.zeros(n, dtype=float, device=device)

        # Bind
        loaded_graph.bind_input("input", new_input)
        loaded_graph.bind_output("output", new_output)

        # Execute
        wp.capture_launch(loaded_graph)
        wp.synchronize_device(device)

        # tmp = 10 * 2 = 20, output = tmp + input = 20 + 10 = 30
        expected = np.ones(n, dtype=np.float32) * 30.0
        np.testing.assert_array_almost_equal(new_output.numpy(), expected)


def test_apic_multiple_internal_allocations(test, device):
    """Test APIC with multiple internal allocations inside graph capture."""
    n = 64

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = os.path.join(tmpdir, "multi_alloc_graph")

        # Only input and final output are bound
        input_data = wp.array(np.ones(n, dtype=np.float32) * 2.0, device=device)
        output_data = wp.zeros(n, dtype=float, device=device)

        with wp.ScopedCapture(device=device, apic=True) as capture:
            # Allocate multiple temporary arrays
            t1 = wp.zeros(n, dtype=float, device=device)
            t2 = wp.zeros(n, dtype=float, device=device)
            t3 = wp.zeros(n, dtype=float, device=device)

            # Multi-stage computation
            wp.launch(scale_kernel, dim=n, inputs=[input_data, t1, 2.0], device=device)  # t1 = 4
            wp.launch(scale_kernel, dim=n, inputs=[t1, t2, 3.0], device=device)  # t2 = 12
            wp.launch(add_kernel, dim=n, inputs=[t1, t2, t3], device=device)  # t3 = 4 + 12 = 16
            wp.launch(add_kernel, dim=n, inputs=[t3, input_data, output_data], device=device)  # out = 16 + 2 = 18

        # Should have 5 memory regions: input, output, t1, t2, t3
        test.assertEqual(len(capture.graph.apic_capture.memory_regions), 5)

        # Verify original graph
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
        np.testing.assert_array_almost_equal(output_data.numpy(), np.ones(n) * 18.0)

        # Save and load
        wp.capture_save(capture.graph, graph_path, inputs={"input": input_data}, outputs={"output": output_data})
        loaded_graph = wp.capture_load(graph_path, device=device)

        # Create new arrays with different values
        new_input = wp.array(np.ones(n, dtype=np.float32) * 5.0, device=device)
        new_output = wp.zeros(n, dtype=float, device=device)

        loaded_graph.bind_input("input", new_input)
        loaded_graph.bind_output("output", new_output)

        wp.capture_launch(loaded_graph)
        wp.synchronize_device(device)

        # t1 = 5 * 2 = 10, t2 = 10 * 3 = 30, t3 = 10 + 30 = 40, output = 40 + 5 = 45
        expected = np.ones(n, dtype=np.float32) * 45.0
        np.testing.assert_array_almost_equal(new_output.numpy(), expected)


class TestApic(unittest.TestCase):
    pass


# Register tests for CUDA devices
devices = get_selected_cuda_test_devices()

add_function_test(TestApic, "test_apic_capture_begin_end", test_apic_capture_begin_end, devices=devices)
add_function_test(TestApic, "test_apic_scoped_capture", test_apic_scoped_capture, devices=devices)
add_function_test(TestApic, "test_apic_multiple_launches", test_apic_multiple_launches, devices=devices)
add_function_test(TestApic, "test_apic_memory_regions", test_apic_memory_regions, devices=devices)
add_function_test(TestApic, "test_apic_save_load_basic", test_apic_save_load_basic, devices=devices)
add_function_test(TestApic, "test_apic_wgf_format", test_apic_wgf_format, devices=devices)
add_function_test(TestApic, "test_apic_capture_class", test_apic_capture_class, devices=devices)
add_function_test(TestApic, "test_apic_array_slicing", test_apic_array_slicing, devices=devices)
add_function_test(TestApic, "test_apic_input_output_bindings", test_apic_input_output_bindings, devices=devices)
add_function_test(TestApic, "test_apic_kernel_info_tracking", test_apic_kernel_info_tracking, devices=devices)
add_function_test(TestApic, "test_apic_launch_record", test_apic_launch_record, devices=devices)
add_function_test(TestApic, "test_apic_graph_execution_unchanged", test_apic_graph_execution_unchanged, devices=devices)
add_function_test(TestApic, "test_apic_save_load_execute", test_apic_save_load_execute, devices=devices)
add_function_test(
    TestApic, "test_apic_load_execute_multiple_kernels", test_apic_load_execute_multiple_kernels, devices=devices
)
add_function_test(TestApic, "test_apic_with_memory_ops", test_apic_with_memory_ops, devices=devices)
add_function_test(TestApic, "test_apic_complex_pipeline", test_apic_complex_pipeline, devices=devices)
add_function_test(TestApic, "test_apic_internal_allocation", test_apic_internal_allocation, devices=devices)
add_function_test(
    TestApic, "test_apic_multiple_internal_allocations", test_apic_multiple_internal_allocations, devices=devices
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
