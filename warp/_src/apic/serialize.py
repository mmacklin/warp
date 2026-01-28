# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Serialization of captured CUDA graphs to WGF format.

Loading is handled by the native C++ implementation in warp.cu.
"""

import ctypes
import os
import struct
from pathlib import Path

from .capture import APICapture, ModuleInfo
from .format import WGFWriter
from .types import (
    APIC_MAX_DIMS,
    APIC_OP_ALLOC,
    APIC_OP_KERNEL_LAUNCH,
    APIC_OP_MEMCPY_D2D,
    APIC_OP_MEMCPY_H2D,
    APIC_OP_MEMSET,
    APIC_PARAM_ARRAY,
    APIC_PARAM_SCALAR,
    APICAllocRecord,
    APICArrayBindingRecord,
    APICLaunchRecord,
    APICMemcpyD2DRecord,
    APICMemcpyH2DRecord,
    APICMemoryRegionRecord,
    APICMemsetRecord,
    APICScalarBindingRecord,
)


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
    from .types import APIC_FORMAT_VERSION

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
        "version": APIC_FORMAT_VERSION,
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
    """Build the memory section containing initial data for internal regions.

    Uses APICMemoryRegionRecord structs for each region.
    """
    chunks = []

    # Count regions with initial data
    regions_with_data = [r for r in capture.memory_regions.values() if r.initial_data is not None]
    chunks.append(struct.pack("<I", len(regions_with_data)))

    # Write each region using the struct
    for region in regions_with_data:
        record = APICMemoryRegionRecord()
        record.region_id = region.region_id
        record.element_size = region.element_size
        record.size = region.size
        record.role = region.role
        record.has_initial_data = 1

        chunks.append(bytes(record))
        chunks.append(region.initial_data)

    return b"".join(chunks)


def _build_operations_section(capture: APICapture) -> bytes:
    """Build the operations section containing launches and memory ops in order.

    Uses ctypes structs for each operation type.
    """
    chunks = []
    total_ops = len(capture.operations)
    chunks.append(struct.pack("<I", total_ops))

    # Write operations in the order they were captured
    for op_type, idx in capture.operations:
        if op_type == "launch":
            chunks.append(_serialize_launch(capture.launches[idx]))
        elif op_type == "memop":
            chunks.append(_serialize_memory_op(capture.memory_ops[idx]))
        elif op_type == "alloc":
            chunks.append(_serialize_alloc(capture.allocs[idx]))

    return b"".join(chunks)


def _serialize_launch(launch) -> bytes:
    """Serialize a kernel launch record using APICLaunchRecord struct."""
    # Encode strings
    key_bytes = launch.kernel_key.encode("utf-8")
    hash_bytes = launch.module_hash.encode("utf-8")

    # Serialize parameter bindings
    param_chunks = []
    for binding in launch.param_bindings:
        param_chunks.append(_serialize_param_binding(binding))
    params_bytes = b"".join(param_chunks)

    # Calculate total size
    total_size = ctypes.sizeof(APICLaunchRecord) + len(key_bytes) + len(hash_bytes) + len(params_bytes)

    # Create the fixed header struct
    record = APICLaunchRecord()
    record.header.op_type = APIC_OP_KERNEL_LAUNCH
    record.header.total_size = total_size
    record.dim = launch.dim
    record.max_blocks = launch.max_blocks
    record.block_dim = launch.block_dim
    record.smem_bytes = launch.smem_bytes
    record.is_forward = 1 if launch.is_forward else 0
    record.kernel_key_len = len(key_bytes)
    record.module_hash_len = len(hash_bytes)
    record.num_params = len(launch.param_bindings)

    return bytes(record) + key_bytes + hash_bytes + params_bytes


def _serialize_param_binding(binding: dict) -> bytes:
    """Serialize a parameter binding using ctypes structs."""
    if binding["type"] == "array":
        record = APICArrayBindingRecord()
        record.type = APIC_PARAM_ARRAY
        record.param_index = binding["param_index"]

        region_id = binding["region_id"] if binding["region_id"] is not None else -1
        record.region_id = region_id
        record.byte_offset = binding["offset"]

        ndim = binding["ndim"]
        record.ndim = ndim
        record.element_size = binding["element_size"]

        # Copy shape and strides (pad with zeros)
        for i in range(APIC_MAX_DIMS):
            if i < ndim:
                record.shape[i] = binding["shape"][i]
                record.strides[i] = binding["strides"][i]
            else:
                record.shape[i] = 0
                record.strides[i] = 0

        return bytes(record)

    else:  # scalar
        record = APICScalarBindingRecord()
        record.type = APIC_PARAM_SCALAR
        record.param_index = binding["param_index"]
        record.size = binding["size"]

        # Copy value bytes
        value_bytes = binding["value"]
        for i, b in enumerate(value_bytes):
            record.value[i] = b

        return bytes(record)


def _serialize_memory_op(op) -> bytes:
    """Serialize a memory operation using ctypes structs."""
    from .capture import MemcpyRecord, MemsetRecord

    if isinstance(op, MemcpyRecord):
        if op.kind == "H2D":
            total_size = ctypes.sizeof(APICMemcpyH2DRecord) + op.size

            record = APICMemcpyH2DRecord()
            record.header.op_type = APIC_OP_MEMCPY_H2D
            record.header.total_size = total_size
            record.dst_region_id = op.dst_region_id
            record.dst_offset = op.dst_offset
            record.size = op.size

            return bytes(record) + op.src_data

        elif op.kind == "D2D":
            record = APICMemcpyD2DRecord()
            record.header.op_type = APIC_OP_MEMCPY_D2D
            record.header.total_size = ctypes.sizeof(APICMemcpyD2DRecord)
            record.dst_region_id = op.dst_region_id
            record.src_region_id = op.src_region_id
            record.dst_offset = op.dst_offset
            record.src_offset = op.src_offset
            record.size = op.size

            return bytes(record)

        # D2H not serialized (output operation)
        return b""

    elif isinstance(op, MemsetRecord):
        record = APICMemsetRecord()
        record.header.op_type = APIC_OP_MEMSET
        record.header.total_size = ctypes.sizeof(APICMemsetRecord)
        record.region_id = op.region_id
        record.value = op.value
        record.offset = op.offset
        record.size = op.size

        return bytes(record)

    return b""


def _serialize_alloc(alloc) -> bytes:
    """Serialize an allocation operation using APICAllocRecord struct."""
    record = APICAllocRecord()
    record.header.op_type = APIC_OP_ALLOC
    record.header.total_size = ctypes.sizeof(APICAllocRecord)
    record.region_id = alloc.region_id
    record.size = alloc.size

    return bytes(record)
