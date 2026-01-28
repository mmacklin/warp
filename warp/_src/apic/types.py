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
    """Kernel launch record (fixed part) - 60 bytes.

    Variable data follows: kernel_key, module_hash, param_bindings[)
    """

    _pack_ = 1
    _fields_ = (
        ("header", APICOpHeader),
        ("dim", c_uint64),
        ("shape", c_int32 * APIC_LAUNCH_MAX_DIMS),
        ("ndim", c_int32),
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
