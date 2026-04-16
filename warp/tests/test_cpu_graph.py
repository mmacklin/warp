# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CPU graph capture and replay using APIC."""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def scale_kernel(input: wp.array(dtype=float), output: wp.array(dtype=float), s: float):
    tid = wp.tid()
    output[tid] = input[tid] * s


@wp.kernel
def add_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), output: wp.array(dtype=float)):
    tid = wp.tid()
    output[tid] = a[tid] + b[tid]


@wp.kernel
def increment_kernel(data: wp.array(dtype=float)):
    tid = wp.tid()
    data[tid] = data[tid] + 1.0


class TestCPUGraph(unittest.TestCase):
    """Tests for CPU graph capture and replay."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_cpu_capture_begin_end(self):
        """Test basic CPU capture lifecycle."""
        device = "cpu"
        wp.capture_begin(device=device)
        graph = wp.capture_end(device=device)
        self.assertIsNotNone(graph)
        self.assertTrue(graph._is_cpu_graph)

    def test_cpu_single_kernel(self):
        """Test capture and replay of a single CPU kernel."""
        n = 100
        device = "cpu"

        a = wp.array(np.ones(n, dtype=np.float32), dtype=float, device=device)
        b = wp.zeros(n, dtype=float, device=device)

        # Capture
        wp.capture_begin(device=device)
        wp.launch(scale_kernel, dim=n, inputs=[a, b, 2.0], device=device)
        graph = wp.capture_end(device=device)

        # Clear output and replay
        b.zero_()
        wp.capture_launch(graph)

        # Verify
        expected = np.ones(n) * 2.0
        np.testing.assert_allclose(b.numpy(), expected)

    def test_cpu_multiple_kernels(self):
        """Test capture and replay of multiple CPU kernels in sequence."""
        n = 64
        device = "cpu"

        a = wp.array(np.ones(n, dtype=np.float32) * 3.0, dtype=float, device=device)
        b = wp.array(np.ones(n, dtype=np.float32) * 5.0, dtype=float, device=device)
        c = wp.zeros(n, dtype=float, device=device)
        d = wp.zeros(n, dtype=float, device=device)

        # Capture: scale a, then add scaled a + b
        wp.capture_begin(device=device)
        wp.launch(scale_kernel, dim=n, inputs=[a, c, 2.0], device=device)
        wp.launch(add_kernel, dim=n, inputs=[c, b, d], device=device)
        graph = wp.capture_end(device=device)

        # Clear outputs and replay
        c.zero_()
        d.zero_()
        wp.capture_launch(graph)

        # Verify: c = a * 2 = 6, d = c + b = 6 + 5 = 11
        np.testing.assert_allclose(c.numpy(), np.ones(n) * 6.0)
        np.testing.assert_allclose(d.numpy(), np.ones(n) * 11.0)

    def test_cpu_replay_multiple_times(self):
        """Test replaying the same CPU graph multiple times."""
        n = 32
        device = "cpu"

        data = wp.zeros(n, dtype=float, device=device)

        # Capture: increment all elements by 1
        wp.capture_begin(device=device)
        wp.launch(increment_kernel, dim=n, inputs=[data], device=device)
        graph = wp.capture_end(device=device)

        # Replay 5 times - each should increment by 1
        for i in range(5):
            wp.capture_launch(graph)

        # After capture (1) + 5 replays = 6 increments
        expected = np.ones(n) * 6.0
        np.testing.assert_allclose(data.numpy(), expected)

    def test_cpu_scoped_capture(self):
        """Test ScopedCapture with CPU device."""
        n = 50
        device = "cpu"

        a = wp.array(np.arange(n, dtype=np.float32), dtype=float, device=device)
        b = wp.zeros(n, dtype=float, device=device)

        with wp.ScopedCapture(device=device) as capture:
            wp.launch(scale_kernel, dim=n, inputs=[a, b, 0.5], device=device)

        # Clear and replay
        b.zero_()
        wp.capture_launch(capture.graph)

        expected = np.arange(n, dtype=np.float32) * 0.5
        np.testing.assert_allclose(b.numpy(), expected)

    def test_cpu_apic_operation_count(self):
        """Test that C++ operation stream records operations."""
        n = 10
        device = "cpu"

        a = wp.zeros(n, dtype=float, device=device)
        b = wp.zeros(n, dtype=float, device=device)

        wp.capture_begin(device=device)
        wp.launch(scale_kernel, dim=n, inputs=[a, b, 1.0], device=device)
        wp.launch(increment_kernel, dim=n, inputs=[b], device=device)
        graph = wp.capture_end(device=device)

        # Check operation count (should be at least 2 kernel launches)
        self.assertIsNotNone(graph.apic)
        self.assertGreaterEqual(graph.apic.operation_count, 2)

    def test_cpu_capture_without_apic(self):
        """Test CPU capture with apic=False (no memory tracking for serialization)."""
        n = 50
        device = "cpu"

        a = wp.array(np.ones(n, dtype=np.float32) * 7.0, dtype=float, device=device)
        b = wp.zeros(n, dtype=float, device=device)

        wp.capture_begin(device=device, apic=False)
        wp.launch(scale_kernel, dim=n, inputs=[a, b, 3.0], device=device)
        graph = wp.capture_end(device=device)

        # Clear and replay
        b.zero_()
        wp.capture_launch(graph)

        expected = np.ones(n) * 21.0
        np.testing.assert_allclose(b.numpy(), expected)


    def test_cpu_strided_copy(self):
        """Test capture and replay with non-contiguous (strided) wp.copy()."""
        n = 20
        device = "cpu"

        # Create a strided source (every other element)
        full = wp.array(np.arange(n, dtype=np.float32), dtype=float, device=device)
        src = full[::2]  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

        # Create a strided destination
        dst_full = wp.zeros(n, dtype=float, device=device)
        dst = dst_full[::2]

        wp.capture_begin(device=device)
        wp.copy(dst, src)
        graph = wp.capture_end(device=device)

        # Clear and replay
        dst_full.zero_()
        wp.capture_launch(graph)

        expected = np.zeros(n, dtype=np.float32)
        expected[::2] = np.arange(0, n, 2, dtype=np.float32)
        np.testing.assert_allclose(dst_full.numpy(), expected)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
