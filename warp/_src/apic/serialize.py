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

"""
Serialization and deserialization of captured CUDA graphs.
"""

import os
import struct
from pathlib import Path

from .capture import APICapture, ModuleInfo
from .format import WGFReader, WGFWriter


def save_graph(capture: APICapture, path: str):
    """
    Save a captured graph to disk.

    Args:
        capture: The APICapture containing recorded operations
        path: Output path for the .wgf file (without extension)

    Creates:
        - {path}.wgf: The main graph file
        - {path}_modules/: Directory containing .cubin files
    """

    base_path = Path(path)
    wgf_path = base_path.with_suffix(".wgf")
    modules_dir = base_path.parent / f"{base_path.stem}_modules"

    # Create modules directory
    modules_dir.mkdir(parents=True, exist_ok=True)

    # Finalize memory data for internal regions
    capture.finalize_memory_data()

    # Export cubin files for each unique module
    for _module_hash, module_info in capture.modules.items():
        cubin_path = modules_dir / module_info.cubin_filename
        _export_module_cubin(module_info, cubin_path)

    # Build metadata
    metadata = _build_metadata(capture)

    # Build memory section
    memory_data = _build_memory_section(capture)

    # Build operations section
    operations_data = _build_operations_section(capture)

    # Write .wgf file
    writer = WGFWriter(str(wgf_path), capture.device.arch)
    writer.add_metadata(metadata)
    writer.add_memory(memory_data)
    writer.add_operations(operations_data)
    writer.write()


def _export_module_cubin(module_info: ModuleInfo, cubin_path: Path):
    """Export a module's cubin file."""
    import glob
    import shutil

    import warp
    import warp._src.context

    # Find the module by name
    user_modules = warp._src.context.user_modules
    module_hash_hex = module_info.module_hash
    module_hash_short = module_hash_hex[:7]  # Warp uses 7 hex chars

    for module in user_modules.values():
        if module.name == module_info.module_name:
            # Try to find the cached binary using the module identifier pattern
            module_name_short = module.get_module_identifier()
            arch = module_info.target_arch

            module_dir = os.path.join(warp.config.kernel_cache_dir, module_name_short)

            if os.path.exists(module_dir):
                # Look for cubin or ptx files matching the architecture
                patterns = [
                    os.path.join(module_dir, f"*.sm{arch}.cubin"),
                    os.path.join(module_dir, f"*.sm{arch}.ptx"),
                ]

                for pattern in patterns:
                    matches = glob.glob(pattern)
                    if matches:
                        # Use the first match
                        shutil.copy2(matches[0], cubin_path)
                        return

            # Fall back to searching all subdirectories
            cache_dir = warp.config.kernel_cache_dir
            for root, _dirs, files in os.walk(cache_dir):
                for f in files:
                    if module_hash_short in f and (f.endswith(f".sm{arch}.cubin") or f.endswith(f".sm{arch}.ptx")):
                        src_path = os.path.join(root, f)
                        shutil.copy2(src_path, cubin_path)
                        return

    raise ValueError(f"Could not find cubin for module {module_info.module_name} ({module_info.module_hash})")


def _build_metadata(capture: APICapture) -> dict:
    """Build the metadata dictionary."""
    # Convert modules
    modules = {}
    for module_hash, info in capture.modules.items():
        modules[module_hash] = {
            "name": info.module_name,
            "cubin_filename": info.cubin_filename,
            "target_arch": info.target_arch,
        }

    # Convert kernels
    kernels = {}
    for kernel_key, info in capture.kernels.items():
        kernels[kernel_key] = {
            "module_hash": info.module_hash,
            "forward_name": info.forward_name,
            "backward_name": info.backward_name,
            "forward_smem_bytes": info.forward_smem_bytes,
            "backward_smem_bytes": info.backward_smem_bytes,
            "block_dim": info.block_dim,
        }

    # Convert memory regions
    regions = {}
    for _base_ptr, region in capture.memory_regions.items():
        regions[str(region.region_id)] = {
            "size": region.size,
            "element_size": region.element_size,
            "role": region.role,
            "has_initial_data": region.initial_data is not None,
        }

    # Input/output bindings
    input_bindings = dict(capture.input_bindings)
    output_bindings = dict(capture.output_bindings)

    return {
        "version": 1,
        "target_arch": capture.device.arch,
        "modules": modules,
        "kernels": kernels,
        "memory_regions": regions,
        "input_bindings": input_bindings,
        "output_bindings": output_bindings,
        "num_launches": len(capture.launches),
        "num_memory_ops": len(capture.memory_ops),
    }


def _build_memory_section(capture: APICapture) -> bytes:
    """Build the memory section containing initial data for internal regions."""
    # Format: [region_count: u32]
    #         [region_id: u32, data_size: u64, data: bytes]...

    chunks = []

    # Count regions with initial data
    regions_with_data = [r for r in capture.memory_regions.values() if r.initial_data is not None]
    chunks.append(struct.pack("<I", len(regions_with_data)))

    # Write each region's data
    for region in regions_with_data:
        chunks.append(struct.pack("<IQ", region.region_id, len(region.initial_data)))
        chunks.append(region.initial_data)

    return b"".join(chunks)


def _build_operations_section(capture: APICapture) -> bytes:
    """Build the operations section containing launches and memory ops in order."""
    # Format: [num_operations: u32]
    #         [op_type: u8, op_data...]...

    chunks = []
    total_ops = len(capture.operations)
    chunks.append(struct.pack("<I", total_ops))

    # Write operations in the order they were captured
    for op_type, idx in capture.operations:
        if op_type == "launch":
            chunks.append(_serialize_launch(capture.launches[idx]))
        elif op_type == "memop":
            chunks.append(_serialize_memory_op(capture.memory_ops[idx]))

    return b"".join(chunks)


def _serialize_launch(launch) -> bytes:
    """Serialize a kernel launch record."""
    # Format: [op_type: u8=1]
    #         [kernel_key_len: u16, kernel_key: str]
    #         [module_hash_len: u16, module_hash: str]
    #         [dim: u64, max_blocks: i32, block_dim: i32, smem_bytes: i32]
    #         [is_forward: u8]
    #         [num_params: u16, param_bindings...]

    chunks = []

    # Op type
    chunks.append(struct.pack("<B", 1))  # KERNEL_LAUNCH

    # Kernel key
    key_bytes = launch.kernel_key.encode("utf-8")
    chunks.append(struct.pack("<H", len(key_bytes)))
    chunks.append(key_bytes)

    # Module hash
    hash_bytes = launch.module_hash.encode("utf-8")
    chunks.append(struct.pack("<H", len(hash_bytes)))
    chunks.append(hash_bytes)

    # Launch parameters
    chunks.append(
        struct.pack(
            "<QiiiB",
            launch.dim,
            launch.max_blocks,
            launch.block_dim,
            launch.smem_bytes,
            1 if launch.is_forward else 0,
        )
    )

    # Parameter bindings
    chunks.append(struct.pack("<H", len(launch.param_bindings)))
    for binding in launch.param_bindings:
        chunks.append(_serialize_param_binding(binding))

    return b"".join(chunks)


def _serialize_param_binding(binding: dict) -> bytes:
    """Serialize a parameter binding."""
    chunks = []

    if binding["type"] == "array":
        # Format: [type: u8=1]
        #         [param_index: u16, region_id: i32 (-1 for null), offset: u64]
        #         [ndim: u8, shape: i64[ndim], strides: i64[ndim]]
        #         [element_size: u32]
        chunks.append(struct.pack("<B", 1))  # ARRAY
        chunks.append(struct.pack("<H", binding["param_index"]))

        region_id = binding["region_id"] if binding["region_id"] is not None else -1
        chunks.append(struct.pack("<iQ", region_id, binding["offset"]))

        ndim = binding["ndim"]
        chunks.append(struct.pack("<B", ndim))
        for s in binding["shape"]:
            chunks.append(struct.pack("<q", s))
        for s in binding["strides"]:
            chunks.append(struct.pack("<q", s))
        chunks.append(struct.pack("<I", binding["element_size"]))

    else:  # scalar
        # Format: [type: u8=2]
        #         [param_index: u16, size: u16, value: bytes]
        chunks.append(struct.pack("<B", 2))  # SCALAR
        chunks.append(struct.pack("<HH", binding["param_index"], binding["size"]))
        chunks.append(binding["value"])

    return b"".join(chunks)


def _serialize_memory_op(op) -> bytes:
    """Serialize a memory operation."""
    from .capture import MemcpyRecord, MemsetRecord

    chunks = []

    if isinstance(op, MemcpyRecord):
        if op.kind == "H2D":
            chunks.append(struct.pack("<B", 2))  # MEMCPY_H2D
            chunks.append(
                struct.pack(
                    "<iQQ",
                    op.dst_region_id,
                    op.dst_offset,
                    op.size,
                )
            )
            # Include source data for H2D
            chunks.append(op.src_data)
        elif op.kind == "D2D":
            chunks.append(struct.pack("<B", 4))  # MEMCPY_D2D
            chunks.append(
                struct.pack(
                    "<iQiQQ",
                    op.dst_region_id,
                    op.dst_offset,
                    op.src_region_id,
                    op.src_offset,
                    op.size,
                )
            )
        # D2H not serialized (output operation)

    elif isinstance(op, MemsetRecord):
        chunks.append(struct.pack("<B", 5))  # MEMSET
        chunks.append(struct.pack("<iQiQ", op.region_id, op.offset, op.value, op.size))

    return b"".join(chunks)


def load_graph(path: str, device=None):
    """
    Load a serialized graph from disk.

    Args:
        path: Path to the .wgf file
        device: Target device (default: current CUDA device)

    Returns:
        A Graph object that can be executed with capture_launch()
    """
    from warp._src.context import Graph

    return Graph.load(path, device)


def load_graph_into(graph, path: str):
    """
    Load a serialized graph from disk into an existing Graph object.

    Args:
        graph: The Graph object to populate
        path: Path to the .wgf file
    """

    base_path = Path(path)
    if not base_path.suffix:
        base_path = base_path.with_suffix(".wgf")

    modules_dir = base_path.parent / f"{base_path.stem}_modules"

    # Read .wgf file
    reader = WGFReader(str(base_path))

    # Verify architecture
    if reader.target_arch != graph.device.arch:
        raise ValueError(
            f"Graph was captured for arch {reader.target_arch}, but target device has arch {graph.device.arch}"
        )

    metadata = reader.get_metadata()
    memory_data = reader.get_memory()
    operations_data = reader.get_operations()

    # Load cubin modules
    loaded_modules = _load_modules(metadata["modules"], modules_dir, graph.device)

    # Allocate memory regions
    memory_regions = _allocate_memory_regions(metadata["memory_regions"], graph.device)

    # Initialize internal regions with saved data
    _initialize_memory_regions(memory_regions, memory_data, graph.device)

    # Parse operations and build execution plan
    launches, memory_ops, operations = _parse_operations(operations_data, metadata, loaded_modules, memory_regions)

    # Populate the graph object
    graph._loaded_modules = loaded_modules
    graph._memory_regions = memory_regions
    graph._launches = launches
    graph._memory_ops = memory_ops
    graph._operations = operations
    graph._metadata = metadata

    # Build input/output region mappings
    for name, region_id in metadata.get("input_bindings", {}).items():
        graph._input_bindings[name] = region_id

    for name, region_id in metadata.get("output_bindings", {}).items():
        graph._output_bindings[name] = region_id

    # Build the CUDA graph by replaying operations during capture
    graph._needs_rebuild = True
    graph._rebuild_cuda_graph()


def _load_modules(modules_metadata: dict, modules_dir: Path, device) -> dict:
    """Load cubin modules from disk."""
    import warp

    loaded = {}
    runtime = warp._src.context.runtime

    for module_hash, info in modules_metadata.items():
        cubin_path = modules_dir / info["cubin_filename"]
        if not cubin_path.exists():
            raise FileNotFoundError(f"Cubin file not found: {cubin_path}")

        # Load the cubin into a CUDA module using file path
        cuda_module = runtime.core.wp_cuda_load_module(device.context, str(cubin_path).encode("utf-8"))
        if cuda_module is None:
            raise RuntimeError(f"Failed to load cubin: {cubin_path}")

        loaded[module_hash] = {
            "cuda_module": cuda_module,
            "info": info,
        }

    return loaded


def _allocate_memory_regions(regions_metadata: dict, device) -> dict:
    """Allocate memory regions on the target device."""
    import warp

    runtime = warp._src.context.runtime
    regions = {}

    for region_id_str, info in regions_metadata.items():
        region_id = int(region_id_str)
        size = info["size"]

        # Allocate device memory
        ptr = runtime.core.wp_alloc_device_default(device.context, size)
        if not ptr:
            raise RuntimeError(f"Failed to allocate {size} bytes on device '{device}'")

        regions[region_id] = {
            "ptr": ptr,
            "size": size,
            "element_size": info["element_size"],
            "role": info["role"],
        }

    return regions


def _initialize_memory_regions(regions: dict, memory_data: bytes, device):
    """Initialize internal memory regions with saved data."""
    import ctypes

    import warp

    if not memory_data:
        return

    offset = 0

    # Read region count
    (region_count,) = struct.unpack_from("<I", memory_data, offset)
    offset += 4

    # Read and initialize each region
    for _ in range(region_count):
        region_id, data_size = struct.unpack_from("<IQ", memory_data, offset)
        offset += 12

        data = memory_data[offset : offset + data_size]
        offset += data_size

        if region_id in regions:
            region = regions[region_id]
            # Copy data to device
            src_ptr = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
            warp._src.context.runtime.core.wp_memcpy_h2d(
                device.context,
                ctypes.c_void_p(region["ptr"]),
                ctypes.cast(src_ptr, ctypes.c_void_p),
                data_size,
                None,
            )


def _parse_operations(operations_data: bytes, metadata: dict, loaded_modules: dict, memory_regions: dict):
    """Parse the operations section and build execution plan preserving order."""
    launches = []
    memory_ops = []
    operations = []  # List of ("launch", idx) or ("memop", idx) in execution order

    if not operations_data:
        return launches, memory_ops, operations

    offset = 0

    # Read operation count
    (num_ops,) = struct.unpack_from("<I", operations_data, offset)
    offset += 4

    for _ in range(num_ops):
        op_type = operations_data[offset]
        offset += 1

        if op_type == 1:  # KERNEL_LAUNCH
            launch, offset = _parse_launch(operations_data, offset, metadata, loaded_modules, memory_regions)
            launches.append(launch)
            operations.append(("launch", len(launches) - 1))
        elif op_type == 2:  # MEMCPY_H2D
            op, offset = _parse_memcpy_h2d(operations_data, offset, memory_regions)
            memory_ops.append(op)
            operations.append(("memop", len(memory_ops) - 1))
        elif op_type == 4:  # MEMCPY_D2D
            op, offset = _parse_memcpy_d2d(operations_data, offset, memory_regions)
            memory_ops.append(op)
            operations.append(("memop", len(memory_ops) - 1))
        elif op_type == 5:  # MEMSET
            op, offset = _parse_memset(operations_data, offset, memory_regions)
            memory_ops.append(op)
            operations.append(("memop", len(memory_ops) - 1))

    return launches, memory_ops, operations


def _parse_launch(data: bytes, offset: int, metadata: dict, loaded_modules: dict, memory_regions: dict):
    """Parse a kernel launch from the operations data."""
    # Kernel key
    (key_len,) = struct.unpack_from("<H", data, offset)
    offset += 2
    kernel_key = data[offset : offset + key_len].decode("utf-8")
    offset += key_len

    # Module hash
    (hash_len,) = struct.unpack_from("<H", data, offset)
    offset += 2
    module_hash = data[offset : offset + hash_len].decode("utf-8")
    offset += hash_len

    # Launch parameters
    dim, max_blocks, block_dim, smem_bytes, is_forward = struct.unpack_from("<QiiiB", data, offset)
    offset += 21

    # Parameter bindings
    (num_params,) = struct.unpack_from("<H", data, offset)
    offset += 2

    param_bindings = []
    for _ in range(num_params):
        binding, offset = _parse_param_binding(data, offset, memory_regions)
        param_bindings.append(binding)

    # Get kernel function from loaded module
    kernel_info = metadata["kernels"][kernel_key]
    kernel_name = kernel_info["forward_name"] if is_forward else kernel_info["backward_name"]

    return {
        "kernel_key": kernel_key,
        "module_hash": module_hash,
        "kernel_name": kernel_name,
        "dim": dim,
        "max_blocks": max_blocks,
        "block_dim": block_dim,
        "smem_bytes": smem_bytes,
        "is_forward": bool(is_forward),
        "param_bindings": param_bindings,
    }, offset


def _parse_param_binding(data: bytes, offset: int, memory_regions: dict):
    """Parse a parameter binding."""
    param_type = data[offset]
    offset += 1

    if param_type == 1:  # ARRAY
        (param_index,) = struct.unpack_from("<H", data, offset)
        offset += 2

        region_id, byte_offset = struct.unpack_from("<iQ", data, offset)
        offset += 12

        (ndim,) = struct.unpack_from("<B", data, offset)
        offset += 1

        shape = []
        for _ in range(ndim):
            (s,) = struct.unpack_from("<q", data, offset)
            shape.append(s)
            offset += 8

        strides = []
        for _ in range(ndim):
            (s,) = struct.unpack_from("<q", data, offset)
            strides.append(s)
            offset += 8

        (element_size,) = struct.unpack_from("<I", data, offset)
        offset += 4

        # Resolve region pointer (store offset for later rebinding)
        ptr = None
        if region_id >= 0 and region_id in memory_regions:
            ptr = memory_regions[region_id]["ptr"] + byte_offset

        return {
            "type": "array",
            "param_index": param_index,
            "region_id": region_id if region_id >= 0 else None,
            "byte_offset": byte_offset,  # Store offset for rebinding
            "ptr": ptr,
            "shape": shape,
            "strides": strides,
            "ndim": ndim,
            "element_size": element_size,
        }, offset

    else:  # SCALAR
        param_index, size = struct.unpack_from("<HH", data, offset)
        offset += 4
        value = data[offset : offset + size]
        offset += size

        return {
            "type": "scalar",
            "param_index": param_index,
            "size": size,
            "value": value,
        }, offset


def _parse_memcpy_h2d(data: bytes, offset: int, memory_regions: dict):
    """Parse a H2D memcpy operation."""
    dst_region_id, dst_offset, size = struct.unpack_from("<iQQ", data, offset)
    offset += 20
    src_data = data[offset : offset + size]
    offset += size

    dst_ptr = None
    if dst_region_id in memory_regions:
        dst_ptr = memory_regions[dst_region_id]["ptr"] + dst_offset

    return {
        "type": "memcpy_h2d",
        "dst_region_id": dst_region_id,
        "dst_offset": dst_offset,
        "dst_ptr": dst_ptr,
        "src_data": src_data,
        "size": size,
    }, offset


def _parse_memcpy_d2d(data: bytes, offset: int, memory_regions: dict):
    """Parse a D2D memcpy operation."""
    # Format: i=4, Q=8, i=4, Q=8, Q=8 = 32 bytes
    dst_region_id, dst_offset, src_region_id, src_offset, size = struct.unpack_from("<iQiQQ", data, offset)
    offset += 32

    dst_ptr = None
    src_ptr = None
    if dst_region_id in memory_regions:
        dst_ptr = memory_regions[dst_region_id]["ptr"] + dst_offset
    if src_region_id in memory_regions:
        src_ptr = memory_regions[src_region_id]["ptr"] + src_offset

    return {
        "type": "memcpy_d2d",
        "dst_region_id": dst_region_id,
        "dst_offset": dst_offset,
        "src_region_id": src_region_id,
        "src_offset": src_offset,
        "dst_ptr": dst_ptr,
        "src_ptr": src_ptr,
        "size": size,
    }, offset


def _parse_memset(data: bytes, offset: int, memory_regions: dict):
    """Parse a memset operation."""
    region_id, mem_offset, value, size = struct.unpack_from("<iQiQ", data, offset)
    offset += 28

    ptr = None
    if region_id in memory_regions:
        ptr = memory_regions[region_id]["ptr"] + mem_offset

    return {
        "type": "memset",
        "region_id": region_id,
        "offset": mem_offset,
        "ptr": ptr,
        "value": value,
        "size": size,
    }, offset
