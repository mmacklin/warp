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

"""Tests for APIC mesh serialization.

These tests verify that wp.Mesh objects can be correctly serialized and
deserialized through APIC, with handle fixup ensuring mesh queries work
correctly after loading.
"""

import os
import tempfile
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices


def create_unit_cube_mesh(device):
    """Create a simple unit cube mesh centered at origin."""
    # 8 vertices of a unit cube centered at origin
    points = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )

    # 12 triangles (2 per face)
    indices = np.array(
        [
            # Front face
            0,
            1,
            2,
            0,
            2,
            3,
            # Back face
            4,
            6,
            5,
            4,
            7,
            6,
            # Left face
            0,
            3,
            7,
            0,
            7,
            4,
            # Right face
            1,
            5,
            6,
            1,
            6,
            2,
            # Bottom face
            0,
            4,
            5,
            0,
            5,
            1,
            # Top face
            3,
            2,
            6,
            3,
            6,
            7,
        ],
        dtype=np.int32,
    )

    mesh_points = wp.array(points, dtype=wp.vec3, device=device)
    mesh_indices = wp.array(indices, dtype=int, device=device)

    mesh = wp.Mesh(points=mesh_points, indices=mesh_indices)
    return mesh


def create_tetrahedron_mesh(device):
    """Create a simple tetrahedron mesh."""
    # 4 vertices of a tetrahedron
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ],
        dtype=np.float32,
    )

    # 4 triangles
    indices = np.array(
        [
            0,
            1,
            2,  # Base
            0,
            1,
            3,  # Side 1
            1,
            2,
            3,  # Side 2
            2,
            0,
            3,  # Side 3
        ],
        dtype=np.int32,
    )

    mesh_points = wp.array(points, dtype=wp.vec3, device=device)
    mesh_indices = wp.array(indices, dtype=int, device=device)

    mesh = wp.Mesh(points=mesh_points, indices=mesh_indices)
    return mesh


# -----------------------------------------------------------------------------
# Kernels for mesh operations
# -----------------------------------------------------------------------------


@wp.kernel
def mesh_query_point_kernel(
    mesh_id: wp.handle,
    query_points: wp.array(dtype=wp.vec3),
    closest_points: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=float),
    faces: wp.array(dtype=int),
):
    """Query closest point on mesh for each input point."""
    tid = wp.tid()

    p = query_points[tid]
    max_dist = 100.0

    face = int(0)
    u = float(0.0)
    v = float(0.0)

    wp.mesh_query_point_no_sign(mesh_id, p, max_dist, face, u, v)

    # Evaluate position on mesh
    cp = wp.mesh_eval_position(mesh_id, face, u, v)

    closest_points[tid] = cp
    distances[tid] = wp.length(cp - p)
    faces[tid] = face


@wp.kernel
def mesh_query_ray_kernel(
    mesh_id: wp.handle,
    ray_origins: wp.array(dtype=wp.vec3),
    ray_dirs: wp.array(dtype=wp.vec3),
    hit_distances: wp.array(dtype=float),
    hit_faces: wp.array(dtype=int),
    hit_flags: wp.array(dtype=int),
):
    """Cast rays against mesh and record hits."""
    tid = wp.tid()

    origin = ray_origins[tid]
    direction = ray_dirs[tid]
    max_t = 100.0

    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    normal = wp.vec3(0.0, 0.0, 0.0)
    face = int(0)

    hit = wp.mesh_query_ray(mesh_id, origin, direction, max_t, t, u, v, sign, normal, face)

    if hit:
        hit_distances[tid] = t
        hit_faces[tid] = face
        hit_flags[tid] = 1
    else:
        hit_distances[tid] = -1.0
        hit_faces[tid] = -1
        hit_flags[tid] = 0


@wp.kernel
def mesh_eval_position_kernel(
    mesh_id: wp.handle,
    face_indices: wp.array(dtype=int),
    bary_u: wp.array(dtype=float),
    bary_v: wp.array(dtype=float),
    positions: wp.array(dtype=wp.vec3),
):
    """Evaluate positions on mesh at given barycentric coordinates."""
    tid = wp.tid()

    face = face_indices[tid]
    u = bary_u[tid]
    v = bary_v[tid]

    pos = wp.mesh_eval_position(mesh_id, face, u, v)
    positions[tid] = pos


@wp.kernel
def mesh_combined_operations_kernel(
    mesh_id: wp.handle,
    query_points: wp.array(dtype=wp.vec3),
    results: wp.array(dtype=float),
):
    """Perform multiple mesh operations and combine results."""
    tid = wp.tid()

    p = query_points[tid]
    max_dist = 100.0

    # Query closest point
    face = int(0)
    u = float(0.0)
    v = float(0.0)
    wp.mesh_query_point_no_sign(mesh_id, p, max_dist, face, u, v)

    # Evaluate position
    cp = wp.mesh_eval_position(mesh_id, face, u, v)
    dist = wp.length(cp - p)

    # Cast ray from point toward origin
    ray_dir = wp.normalize(wp.vec3(0.0, 0.0, 0.0) - p)
    t = float(0.0)
    ru = float(0.0)
    rv = float(0.0)
    sign = float(0.0)
    normal = wp.vec3(0.0, 0.0, 0.0)
    ray_face = int(0)

    hit = wp.mesh_query_ray(mesh_id, p, ray_dir, max_dist, t, ru, rv, sign, normal, ray_face)

    # Combine results
    if hit:
        results[tid] = dist + t
    else:
        results[tid] = dist


# -----------------------------------------------------------------------------
# Test functions
# -----------------------------------------------------------------------------


def apic_mesh_query_point(test, device):
    """Test mesh point queries through APIC save/load."""
    mesh = create_unit_cube_mesh(device)
    n = 8

    # Query points outside the cube
    query_pts = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
        ],
        dtype=np.float32,
    )

    query_points = wp.array(query_pts, dtype=wp.vec3, device=device)
    closest_points = wp.zeros(n, dtype=wp.vec3, device=device)
    distances = wp.zeros(n, dtype=float, device=device)
    faces = wp.zeros(n, dtype=int, device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = os.path.join(tmpdir, "mesh_query_point_graph")

        # Capture mesh query operation
        with wp.ScopedCapture(device=device, apic=True) as capture:
            wp.launch(
                mesh_query_point_kernel,
                dim=n,
                inputs=[mesh.id, query_points, closest_points, distances, faces],
                device=device,
            )

        # Execute original to get reference results
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)

        ref_closest = closest_points.numpy().copy()
        ref_distances = distances.numpy().copy()
        ref_faces = faces.numpy().copy()

        # Verify some basic properties
        # Point at (1,0,0) should be 0.5 units from cube face
        test.assertAlmostEqual(ref_distances[0], 0.5, places=4)

        # Reset outputs
        closest_points.zero_()
        distances.zero_()
        faces.zero_()

        # Save graph
        wp.capture_save(
            capture.graph,
            graph_path,
            inputs={"query_points": query_points},
            outputs={
                "closest_points": closest_points,
                "distances": distances,
                "faces": faces,
            },
        )

        # Load graph
        loaded_graph = wp.capture_load(graph_path, device=device)

        # Create new output arrays
        new_closest = wp.zeros(n, dtype=wp.vec3, device=device)
        new_distances = wp.zeros(n, dtype=float, device=device)
        new_faces = wp.zeros(n, dtype=int, device=device)

        # Execute loaded graph
        wp.capture_launch(loaded_graph)
        wp.synchronize_device(device)

        # Get outputs
        loaded_graph.get_param("closest_points", new_closest)
        loaded_graph.get_param("distances", new_distances)
        loaded_graph.get_param("faces", new_faces)

        # Verify results match
        # Note: face indices may differ between original and loaded mesh due to BVH construction
        # being slightly different, but the actual closest points and distances should match
        np.testing.assert_array_almost_equal(new_closest.numpy(), ref_closest, decimal=4)
        np.testing.assert_array_almost_equal(new_distances.numpy(), ref_distances, decimal=4)


def apic_mesh_query_ray(test, device):
    """Test mesh ray queries through APIC save/load."""
    mesh = create_unit_cube_mesh(device)
    n = 6

    # Rays pointing at the cube from outside
    origins = np.array(
        [
            [2.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, -2.0],
        ],
        dtype=np.float32,
    )

    # Directions pointing toward center
    dirs = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    ray_origins = wp.array(origins, dtype=wp.vec3, device=device)
    ray_dirs = wp.array(dirs, dtype=wp.vec3, device=device)
    hit_distances = wp.zeros(n, dtype=float, device=device)
    hit_faces = wp.zeros(n, dtype=int, device=device)
    hit_flags = wp.zeros(n, dtype=int, device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = os.path.join(tmpdir, "mesh_query_ray_graph")

        # Capture mesh ray query operation
        with wp.ScopedCapture(device=device, apic=True) as capture:
            wp.launch(
                mesh_query_ray_kernel,
                dim=n,
                inputs=[mesh.id, ray_origins, ray_dirs, hit_distances, hit_faces, hit_flags],
                device=device,
            )

        # Execute original to get reference results
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)

        ref_distances = hit_distances.numpy().copy()
        ref_faces = hit_faces.numpy().copy()
        ref_flags = hit_flags.numpy().copy()

        # All rays should hit (flag = 1)
        test.assertTrue(np.all(ref_flags == 1))
        # Distance should be 1.5 (from 2.0 to 0.5 cube face)
        np.testing.assert_array_almost_equal(ref_distances, np.full(n, 1.5), decimal=4)

        # Reset outputs
        hit_distances.zero_()
        hit_faces.zero_()
        hit_flags.zero_()

        # Save graph
        wp.capture_save(
            capture.graph,
            graph_path,
            inputs={
                "ray_origins": ray_origins,
                "ray_dirs": ray_dirs,
            },
            outputs={
                "hit_distances": hit_distances,
                "hit_faces": hit_faces,
                "hit_flags": hit_flags,
            },
        )

        # Load graph
        loaded_graph = wp.capture_load(graph_path, device=device)

        # Create new output arrays
        new_distances = wp.zeros(n, dtype=float, device=device)
        new_faces = wp.zeros(n, dtype=int, device=device)
        new_flags = wp.zeros(n, dtype=int, device=device)

        # Execute loaded graph
        wp.capture_launch(loaded_graph)
        wp.synchronize_device(device)

        # Get outputs
        loaded_graph.get_param("hit_distances", new_distances)
        loaded_graph.get_param("hit_faces", new_faces)
        loaded_graph.get_param("hit_flags", new_flags)

        # Verify results match
        # Note: face indices may differ between original and loaded mesh due to BVH construction
        # being slightly different, but distances and hit flags should match
        np.testing.assert_array_almost_equal(new_distances.numpy(), ref_distances, decimal=4)
        np.testing.assert_array_equal(new_flags.numpy(), ref_flags)


def apic_mesh_eval_position(test, device):
    """Test mesh position evaluation through APIC save/load."""
    mesh = create_tetrahedron_mesh(device)
    n = 4

    # Face indices and barycentric coordinates
    face_indices = wp.array([0, 0, 1, 2], dtype=int, device=device)
    bary_u = wp.array([1.0, 0.0, 0.5, 0.33], dtype=float, device=device)
    bary_v = wp.array([0.0, 1.0, 0.25, 0.33], dtype=float, device=device)
    positions = wp.zeros(n, dtype=wp.vec3, device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = os.path.join(tmpdir, "mesh_eval_position_graph")

        # Capture mesh eval operation
        with wp.ScopedCapture(device=device, apic=True) as capture:
            wp.launch(
                mesh_eval_position_kernel,
                dim=n,
                inputs=[mesh.id, face_indices, bary_u, bary_v, positions],
                device=device,
            )

        # Execute original to get reference results
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)

        ref_positions = positions.numpy().copy()

        # Reset output
        positions.zero_()

        # Save graph
        wp.capture_save(
            capture.graph,
            graph_path,
            inputs={
                "face_indices": face_indices,
                "bary_u": bary_u,
                "bary_v": bary_v,
            },
            outputs={"positions": positions},
        )

        # Load graph
        loaded_graph = wp.capture_load(graph_path, device=device)

        # Create new output array
        new_positions = wp.zeros(n, dtype=wp.vec3, device=device)

        # Execute loaded graph
        wp.capture_launch(loaded_graph)
        wp.synchronize_device(device)

        # Get output
        loaded_graph.get_param("positions", new_positions)

        # Verify results match
        np.testing.assert_array_almost_equal(new_positions.numpy(), ref_positions, decimal=4)


def apic_mesh_combined_operations(test, device):
    """Test combined mesh operations through APIC save/load."""
    mesh = create_unit_cube_mesh(device)
    n = 8

    # Query points at various positions
    query_pts = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
            [-0.5, -0.5, -0.5],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )

    query_points = wp.array(query_pts, dtype=wp.vec3, device=device)
    results = wp.zeros(n, dtype=float, device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = os.path.join(tmpdir, "mesh_combined_graph")

        # Capture combined mesh operations
        with wp.ScopedCapture(device=device, apic=True) as capture:
            wp.launch(
                mesh_combined_operations_kernel,
                dim=n,
                inputs=[mesh.id, query_points, results],
                device=device,
            )

        # Execute original to get reference results
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)

        ref_results = results.numpy().copy()

        # Reset output
        results.zero_()

        # Save graph
        wp.capture_save(
            capture.graph,
            graph_path,
            inputs={"query_points": query_points},
            outputs={"results": results},
        )

        # Load graph
        loaded_graph = wp.capture_load(graph_path, device=device)

        # Create new output array
        new_results = wp.zeros(n, dtype=float, device=device)

        # Execute loaded graph
        wp.capture_launch(loaded_graph)
        wp.synchronize_device(device)

        # Get output
        loaded_graph.get_param("results", new_results)

        # Verify results match
        np.testing.assert_array_almost_equal(new_results.numpy(), ref_results, decimal=4)


def apic_mesh_handle_in_struct(test, device):
    """Test mesh handle stored in a struct array through APIC save/load."""

    @wp.struct
    class MeshQuery:
        mesh_id: wp.handle
        query_point: wp.vec3
        max_dist: float

    @wp.kernel
    def query_from_struct_kernel(
        queries: wp.array(dtype=MeshQuery),
        distances: wp.array(dtype=float),
    ):
        tid = wp.tid()

        q = queries[tid]
        face = int(0)
        u = float(0.0)
        v = float(0.0)

        wp.mesh_query_point_no_sign(q.mesh_id, q.query_point, q.max_dist, face, u, v)
        cp = wp.mesh_eval_position(q.mesh_id, face, u, v)
        distances[tid] = wp.length(cp - q.query_point)

    mesh = create_unit_cube_mesh(device)
    n = 4

    # Create struct array with queries
    queries = wp.zeros(n, dtype=MeshQuery, device=device)
    distances = wp.zeros(n, dtype=float, device=device)

    # Initialize queries on host
    queries_host = queries.numpy()
    query_pts = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ]

    for i in range(n):
        queries_host[i]["mesh_id"] = mesh.id
        queries_host[i]["query_point"] = query_pts[i]
        queries_host[i]["max_dist"] = 100.0

    wp.copy(queries, wp.array(queries_host, dtype=MeshQuery, device=device))
    wp.synchronize_device(device)

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = os.path.join(tmpdir, "mesh_struct_graph")

        # Capture kernel with struct containing mesh handle
        with wp.ScopedCapture(device=device, apic=True) as capture:
            wp.launch(
                query_from_struct_kernel,
                dim=n,
                inputs=[queries, distances],
                device=device,
            )

        # Execute original to get reference results
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)

        ref_distances = distances.numpy().copy()

        # Verify expected distances (0.5 for axis-aligned points, ~0.366 for corner)
        test.assertAlmostEqual(ref_distances[0], 0.5, places=3)
        test.assertAlmostEqual(ref_distances[1], 0.5, places=3)
        test.assertAlmostEqual(ref_distances[2], 0.5, places=3)

        # Reset output
        distances.zero_()

        # Save graph
        wp.capture_save(
            capture.graph,
            graph_path,
            inputs={"queries": queries},
            outputs={"distances": distances},
        )

        # Load graph
        loaded_graph = wp.capture_load(graph_path, device=device)

        # Create new output array
        new_distances = wp.zeros(n, dtype=float, device=device)

        # Execute loaded graph
        wp.capture_launch(loaded_graph)
        wp.synchronize_device(device)

        # Get output
        loaded_graph.get_param("distances", new_distances)

        # Verify results match
        np.testing.assert_array_almost_equal(new_distances.numpy(), ref_distances, decimal=4)


def apic_multiple_meshes(test, device):
    """Test multiple meshes through APIC save/load."""

    @wp.kernel
    def query_multiple_meshes_kernel(
        mesh1_id: wp.handle,
        mesh2_id: wp.handle,
        query_points: wp.array(dtype=wp.vec3),
        dist1: wp.array(dtype=float),
        dist2: wp.array(dtype=float),
    ):
        tid = wp.tid()

        p = query_points[tid]
        max_dist = 100.0

        # Query mesh 1
        face1 = int(0)
        u1 = float(0.0)
        v1 = float(0.0)
        wp.mesh_query_point_no_sign(mesh1_id, p, max_dist, face1, u1, v1)
        cp1 = wp.mesh_eval_position(mesh1_id, face1, u1, v1)
        dist1[tid] = wp.length(cp1 - p)

        # Query mesh 2
        face2 = int(0)
        u2 = float(0.0)
        v2 = float(0.0)
        wp.mesh_query_point_no_sign(mesh2_id, p, max_dist, face2, u2, v2)
        cp2 = wp.mesh_eval_position(mesh2_id, face2, u2, v2)
        dist2[tid] = wp.length(cp2 - p)

    mesh1 = create_unit_cube_mesh(device)
    mesh2 = create_tetrahedron_mesh(device)
    n = 4

    query_pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    query_points = wp.array(query_pts, dtype=wp.vec3, device=device)
    dist1 = wp.zeros(n, dtype=float, device=device)
    dist2 = wp.zeros(n, dtype=float, device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = os.path.join(tmpdir, "multiple_meshes_graph")

        # Capture kernel with multiple meshes
        with wp.ScopedCapture(device=device, apic=True) as capture:
            wp.launch(
                query_multiple_meshes_kernel,
                dim=n,
                inputs=[mesh1.id, mesh2.id, query_points, dist1, dist2],
                device=device,
            )

        # Execute original to get reference results
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)

        ref_dist1 = dist1.numpy().copy()
        ref_dist2 = dist2.numpy().copy()

        # Reset outputs
        dist1.zero_()
        dist2.zero_()

        # Save graph
        wp.capture_save(
            capture.graph,
            graph_path,
            inputs={"query_points": query_points},
            outputs={"dist1": dist1, "dist2": dist2},
        )

        # Load graph
        loaded_graph = wp.capture_load(graph_path, device=device)

        # Create new output arrays
        new_dist1 = wp.zeros(n, dtype=float, device=device)
        new_dist2 = wp.zeros(n, dtype=float, device=device)

        # Execute loaded graph
        wp.capture_launch(loaded_graph)
        wp.synchronize_device(device)

        # Get outputs
        loaded_graph.get_param("dist1", new_dist1)
        loaded_graph.get_param("dist2", new_dist2)

        # Verify results match
        np.testing.assert_array_almost_equal(new_dist1.numpy(), ref_dist1, decimal=4)
        np.testing.assert_array_almost_equal(new_dist2.numpy(), ref_dist2, decimal=4)


# -----------------------------------------------------------------------------
# Test registration
# -----------------------------------------------------------------------------


class TestApicMesh(unittest.TestCase):
    pass


# Register tests for CUDA devices
devices = get_selected_cuda_test_devices()

add_function_test(TestApicMesh, "test_apic_mesh_query_point", apic_mesh_query_point, devices=devices)
add_function_test(TestApicMesh, "test_apic_mesh_query_ray", apic_mesh_query_ray, devices=devices)
add_function_test(TestApicMesh, "test_apic_mesh_eval_position", apic_mesh_eval_position, devices=devices)
add_function_test(TestApicMesh, "test_apic_mesh_combined_operations", apic_mesh_combined_operations, devices=devices)
add_function_test(TestApicMesh, "test_apic_mesh_handle_in_struct", apic_mesh_handle_in_struct, devices=devices)
add_function_test(TestApicMesh, "test_apic_multiple_meshes", apic_multiple_meshes, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
