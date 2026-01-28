# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
APIC serialization types - ctypes structures that mirror apic_types.h

These structures are used for binary serialization of APIC records.
They must match the C structs in warp/native/apic_types.h exactly.
"""

import ctypes
from ctypes import Structure, c_int32, c_int64, c_uint8, c_uint16, c_uint32, c_uint64

# =============================================================================
# Constants (must match apic_types.h)
# =============================================================================

APIC_FORMAT_VERSION = 2
APIC_MAGIC = b"WGF1"
APIC_MAX_DIMS = 4
APIC_MAX_SCALAR_SIZE = 128
APIC_LAUNCH_MAX_DIMS = 4

# Operation types
APIC_OP_KERNEL_LAUNCH = 1
APIC_OP_MEMCPY_H2D = 2
APIC_OP_MEMCPY_D2H = 3
APIC_OP_MEMCPY_D2D = 4
APIC_OP_MEMSET = 5
APIC_OP_ALLOC = 6

# Parameter binding types
APIC_PARAM_ARRAY = 1
APIC_PARAM_SCALAR = 2

# Memory region roles
APIC_ROLE_INTERNAL = 0
APIC_ROLE_INPUT = 1
APIC_ROLE_OUTPUT = 2
APIC_ROLE_INPUT_OUTPUT = 3

# Section types
APIC_SECTION_METADATA = 0x01
APIC_SECTION_MEMORY = 0x02
APIC_SECTION_OPERATIONS = 0x03


# =============================================================================
# WGF File Header
# =============================================================================


class APICFileHeader(Structure):
    """WGF file header - 64 bytes."""

    _pack_ = 1
    _fields_ = (
        ("magic", c_uint8 * 4),
        ("version", c_uint32),
        ("flags", c_uint32),
        ("num_sections", c_uint32),
        ("section_table_offset", c_uint64),
        ("target_arch", c_uint32),
        ("_reserved", c_uint32 * 9),
    )


class APICSectionEntry(Structure):
    """Section table entry - 32 bytes."""

    _pack_ = 1
    _fields_ = (
        ("type", c_uint32),
        ("flags", c_uint32),
        ("offset", c_uint64),
        ("size", c_int64),
        ("uncompressed_size", c_int64),
    )


# =============================================================================
# Operation Records
# =============================================================================


class APICOpHeader(Structure):
    """Common header for all operations - 8 bytes."""

    _pack_ = 1
    _fields_ = (
        ("op_type", c_uint8),
        ("_pad", c_uint8 * 3),
        ("total_size", c_uint32),
    )


class APICLaunchRecord(Structure):
    """Kernel launch record (fixed part) - 40 bytes.

    Variable data follows: kernel_key, module_hash, param_bindings[)
    """

    _pack_ = 1
    _fields_ = (
        ("header", APICOpHeader),
        ("dim", c_uint64),
        ("max_blocks", c_int32),
        ("block_dim", c_int32),
        ("smem_bytes", c_int32),
        ("is_forward", c_uint8),
        ("_pad1", c_uint8 * 3),
        ("kernel_key_len", c_uint16),
        ("module_hash_len", c_uint16),
        ("num_params", c_uint16),
        ("_pad2", c_uint16),
    )


class APICArrayBindingRecord(Structure):
    """Array parameter binding - 88 bytes."""

    _pack_ = 1
    _fields_ = (
        ("type", c_uint8),
        ("ndim", c_uint8),
        ("param_index", c_uint16),
        ("region_id", c_int32),
        ("byte_offset", c_uint64),
        ("shape", c_int64 * APIC_MAX_DIMS),
        ("strides", c_int64 * APIC_MAX_DIMS),
        ("element_size", c_uint32),
        ("_pad", c_uint32),
    )


class APICScalarBindingRecord(Structure):
    """Scalar parameter binding - 136 bytes."""

    _pack_ = 1
    _fields_ = (
        ("type", c_uint8),
        ("_pad1", c_uint8),
        ("param_index", c_uint16),
        ("size", c_uint16),
        ("_pad2", c_uint16),
        ("value", c_uint8 * APIC_MAX_SCALAR_SIZE),
    )


class APICMemcpyH2DRecord(Structure):
    """Memcpy Host-to-Device record - 32 bytes fixed.

    Variable data follows: uint8_t data[size)
    """

    _pack_ = 1
    _fields_ = (
        ("header", APICOpHeader),
        ("dst_region_id", c_int32),
        ("_pad", c_uint32),
        ("dst_offset", c_uint64),
        ("size", c_uint64),
    )


class APICMemcpyD2DRecord(Structure):
    """Memcpy Device-to-Device record - 40 bytes."""

    _pack_ = 1
    _fields_ = (
        ("header", APICOpHeader),
        ("dst_region_id", c_int32),
        ("src_region_id", c_int32),
        ("dst_offset", c_uint64),
        ("src_offset", c_uint64),
        ("size", c_uint64),
    )


class APICMemsetRecord(Structure):
    """Memset record - 32 bytes."""

    _pack_ = 1
    _fields_ = (
        ("header", APICOpHeader),
        ("region_id", c_int32),
        ("value", c_int32),
        ("offset", c_uint64),
        ("size", c_uint64),
    )


class APICAllocRecord(Structure):
    """In-graph allocation record - 24 bytes."""

    _pack_ = 1
    _fields_ = (
        ("header", APICOpHeader),
        ("region_id", c_int32),
        ("_pad", c_uint32),
        ("size", c_uint64),
    )


# =============================================================================
# Memory Section Records
# =============================================================================


class APICMemoryRegionRecord(Structure):
    """Memory region record - 24 bytes fixed.

    If has_initial_data: uint8_t initial_data[size] follows
    """

    _pack_ = 1
    _fields_ = (
        ("region_id", c_uint32),
        ("element_size", c_uint32),
        ("size", c_uint64),
        ("role", c_uint8),
        ("has_initial_data", c_uint8),
        ("_pad", c_uint8 * 6),
    )


class APICBindingRecord(Structure):
    """Binding record (name -> region_id) - 8 bytes fixed.

    Variable data follows: char name[name_len)
    """

    _pack_ = 1
    _fields_ = (
        ("region_id", c_uint32),
        ("name_len", c_uint16),
        ("_pad", c_uint8 * 2),
    )


# =============================================================================
# Helper Functions
# =============================================================================


def struct_to_bytes(struct: Structure) -> bytes:
    """Convert a ctypes Structure to bytes."""
    return bytes(struct)


def create_launch_record(
    dim: int,
    max_blocks: int,
    block_dim: int,
    smem_bytes: int,
    is_forward: bool,
    kernel_key: str,
    module_hash: str,
    param_bindings: list[bytes],
) -> bytes:
    """Create a serialized launch record with variable data.

    Args:
        dim: Total thread count
        max_blocks: Maximum blocks
        block_dim: Threads per block
        smem_bytes: Shared memory bytes
        is_forward: True for forward pass
        kernel_key: Kernel identifier string
        module_hash: Module hash string
        param_bindings: List of serialized param binding bytes

    Returns:
        Complete serialized launch record bytes
    """
    key_bytes = kernel_key.encode("utf-8")
    hash_bytes = module_hash.encode("utf-8")
    params_bytes = b"".join(param_bindings)

    # Calculate total size
    total_size = ctypes.sizeof(APICLaunchRecord) + len(key_bytes) + len(hash_bytes) + len(params_bytes)

    # Create the fixed header
    record = APICLaunchRecord()
    record.header.op_type = APIC_OP_KERNEL_LAUNCH
    record.header.total_size = total_size
    record.dim = dim
    record.max_blocks = max_blocks
    record.block_dim = block_dim
    record.smem_bytes = smem_bytes
    record.is_forward = 1 if is_forward else 0
    record.kernel_key_len = len(key_bytes)
    record.module_hash_len = len(hash_bytes)
    record.num_params = len(param_bindings)

    return bytes(record) + key_bytes + hash_bytes + params_bytes


def create_array_binding(
    param_index: int,
    region_id: int,
    byte_offset: int,
    ndim: int,
    shape: tuple,
    strides: tuple,
    element_size: int,
) -> bytes:
    """Create a serialized array binding record.

    Args:
        param_index: Parameter index in kernel signature
        region_id: Memory region ID (-1 for null)
        byte_offset: Byte offset within region
        ndim: Number of dimensions
        shape: Array shape tuple
        strides: Array strides tuple
        element_size: Element size in bytes

    Returns:
        Serialized array binding bytes
    """
    record = APICArrayBindingRecord()
    record.type = APIC_PARAM_ARRAY
    record.ndim = ndim
    record.param_index = param_index
    record.region_id = region_id
    record.byte_offset = byte_offset
    record.element_size = element_size

    for i in range(APIC_MAX_DIMS):
        record.shape[i] = shape[i] if i < ndim else 0
        record.strides[i] = strides[i] if i < ndim else 0

    return bytes(record)


def create_scalar_binding(param_index: int, value_bytes: bytes) -> bytes:
    """Create a serialized scalar binding record.

    Args:
        param_index: Parameter index in kernel signature
        value_bytes: Raw scalar value bytes

    Returns:
        Serialized scalar binding bytes
    """
    if len(value_bytes) > APIC_MAX_SCALAR_SIZE:
        raise ValueError(f"Scalar value too large: {len(value_bytes)} > {APIC_MAX_SCALAR_SIZE}")

    record = APICScalarBindingRecord()
    record.type = APIC_PARAM_SCALAR
    record.param_index = param_index
    record.size = len(value_bytes)

    for i, b in enumerate(value_bytes):
        record.value[i] = b

    return bytes(record)


def create_memcpy_h2d_record(dst_region_id: int, dst_offset: int, data: bytes) -> bytes:
    """Create a serialized memcpy H2D record with inline data.

    Args:
        dst_region_id: Destination region ID
        dst_offset: Byte offset in destination
        data: Source data bytes

    Returns:
        Serialized memcpy H2D record bytes
    """
    total_size = ctypes.sizeof(APICMemcpyH2DRecord) + len(data)

    record = APICMemcpyH2DRecord()
    record.header.op_type = APIC_OP_MEMCPY_H2D
    record.header.total_size = total_size
    record.dst_region_id = dst_region_id
    record.dst_offset = dst_offset
    record.size = len(data)

    return bytes(record) + data


def create_memcpy_d2d_record(
    dst_region_id: int, dst_offset: int, src_region_id: int, src_offset: int, size: int
) -> bytes:
    """Create a serialized memcpy D2D record.

    Args:
        dst_region_id: Destination region ID
        dst_offset: Byte offset in destination
        src_region_id: Source region ID
        src_offset: Byte offset in source
        size: Number of bytes to copy

    Returns:
        Serialized memcpy D2D record bytes
    """
    record = APICMemcpyD2DRecord()
    record.header.op_type = APIC_OP_MEMCPY_D2D
    record.header.total_size = ctypes.sizeof(APICMemcpyD2DRecord)
    record.dst_region_id = dst_region_id
    record.src_region_id = src_region_id
    record.dst_offset = dst_offset
    record.src_offset = src_offset
    record.size = size

    return bytes(record)


def create_memset_record(region_id: int, offset: int, value: int, size: int) -> bytes:
    """Create a serialized memset record.

    Args:
        region_id: Target region ID
        offset: Byte offset in region
        value: Value to set
        size: Number of bytes to set

    Returns:
        Serialized memset record bytes
    """
    record = APICMemsetRecord()
    record.header.op_type = APIC_OP_MEMSET
    record.header.total_size = ctypes.sizeof(APICMemsetRecord)
    record.region_id = region_id
    record.value = value
    record.offset = offset
    record.size = size

    return bytes(record)


def create_alloc_record(region_id: int, size: int) -> bytes:
    """Create a serialized allocation record.

    Args:
        region_id: Region ID for the allocation
        size: Number of bytes to allocate

    Returns:
        Serialized alloc record bytes
    """
    record = APICAllocRecord()
    record.header.op_type = APIC_OP_ALLOC
    record.header.total_size = ctypes.sizeof(APICAllocRecord)
    record.region_id = region_id
    record.size = size

    return bytes(record)


def create_memory_region_record(
    region_id: int,
    element_size: int,
    size: int,
    role: int,
    initial_data: bytes | None = None,
) -> bytes:
    """Create a serialized memory region record.

    Args:
        region_id: Region identifier
        element_size: Element size in bytes
        size: Total size in bytes
        role: Memory role (APIC_ROLE_*)
        initial_data: Initial data bytes (or None)

    Returns:
        Serialized memory region record bytes
    """
    record = APICMemoryRegionRecord()
    record.region_id = region_id
    record.element_size = element_size
    record.size = size
    record.role = role
    record.has_initial_data = 1 if initial_data else 0

    result = bytes(record)
    if initial_data:
        result += initial_data

    return result
