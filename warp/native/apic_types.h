/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// APIC Format Constants
// =============================================================================

#define APIC_FORMAT_VERSION 2
#define APIC_MAGIC "WGF1"
#define APIC_MAGIC_VALUE 0x31464757  // "WGF1" as little-endian uint32

// Maximum array dimensions (matches Warp's ARRAY_MAX_DIMS)
#define APIC_MAX_DIMS 4

// Maximum scalar parameter size (covers mat44d = 128 bytes)
#define APIC_MAX_SCALAR_SIZE 128

// Maximum launch dimensions (must match LAUNCH_MAX_DIMS in builtin.h)
#define APIC_LAUNCH_MAX_DIMS 4

// =============================================================================
// Enums
// =============================================================================

// Operation types
typedef enum {
    APIC_OP_KERNEL_LAUNCH = 1,
    APIC_OP_MEMCPY_H2D = 2,
    APIC_OP_MEMCPY_D2H = 3,  // Reserved for future
    APIC_OP_MEMCPY_D2D = 4,
    APIC_OP_MEMSET = 5,
    APIC_OP_ALLOC = 6,  // In-graph allocation

    // Future: high-level Warp operations
    // APIC_OP_MESH_CREATE = 10,
    // APIC_OP_VOLUME_CREATE = 11,
    // APIC_OP_BVH_CREATE = 12,
} APICOpType;

// Parameter binding types
typedef enum {
    APIC_PARAM_ARRAY = 1,
    APIC_PARAM_SCALAR = 2,
} APICParamType;

// Memory region roles
typedef enum {
    APIC_ROLE_INTERNAL = 0,
    APIC_ROLE_INPUT = 1,
    APIC_ROLE_OUTPUT = 2,
    APIC_ROLE_INPUT_OUTPUT = 3,
} APICMemoryRole;

// =============================================================================
// WGF File Header
// =============================================================================

// All serialization structs are packed to ensure binary compatibility with Python
#pragma pack(push, 1)

typedef struct {
    uint8_t magic[4];  // "WGF1"
    uint32_t version;  // APIC_FORMAT_VERSION
    uint32_t flags;  // Reserved flags
    uint32_t num_sections;  // Number of sections
    uint64_t section_table_offset;  // Offset to section table
    uint32_t target_arch;  // CUDA SM version (e.g., 86)
    uint32_t _reserved[9];  // Reserved for future use
} APICFileHeader;  // 64 bytes

// Section types
typedef enum {
    APIC_SECTION_METADATA = 0x01,
    APIC_SECTION_MEMORY = 0x02,
    APIC_SECTION_OPERATIONS = 0x03,
} APICSectionType;

typedef struct {
    uint32_t type;  // APICSectionType
    uint32_t flags;  // Section flags
    uint64_t offset;  // Offset from file start
    int64_t size;  // Section size (compressed)
    int64_t uncompressed_size;  // Uncompressed size
} APICSectionEntry;  // 32 bytes

// =============================================================================
// Operation Records
// =============================================================================

// Common header for all operations
typedef struct {
    uint8_t op_type;  // APICOpType
    uint8_t _pad[3];  // Padding for alignment
    uint32_t total_size;  // Total bytes including header and variable data
} APICOpHeader;  // 8 bytes

// -----------------------------------------------------------------------------
// Kernel Launch
// -----------------------------------------------------------------------------

// Kernel launch record (fixed part)
// Variable data follows: kernel_key, module_hash, param_bindings[]
// Note: shape/ndim are stored in the launch_bounds_t which is param[0]
typedef struct {
    APICOpHeader header;  // op_type = APIC_OP_KERNEL_LAUNCH

    // Launch parameters
    uint64_t dim;  // Total threads
    int32_t max_blocks;  // Maximum blocks
    int32_t block_dim;  // Threads per block
    int32_t smem_bytes;  // Shared memory bytes
    uint8_t is_forward;  // 1 for forward pass, 0 for backward
    uint8_t _pad1[3];

    // Variable data sizes
    uint16_t kernel_key_len;  // Length of kernel_key string
    uint16_t module_hash_len;  // Length of module_hash string
    uint16_t num_params;  // Number of parameter bindings
    uint16_t _pad2;

    // Variable data follows in order:
    // 1. char kernel_key[kernel_key_len]
    // 2. char module_hash[module_hash_len]
    // 3. Parameter bindings (APICArrayBindingRecord or APICScalarBindingRecord)
    //    - param[0] is always launch_bounds_t (contains shape/ndim/size)
} APICLaunchRecord;  // 40 bytes fixed

// Array parameter binding (fixed size)
typedef struct {
    uint8_t type;  // APIC_PARAM_ARRAY
    uint8_t ndim;  // Number of dimensions
    uint16_t param_index;  // Parameter index in kernel signature
    int32_t region_id;  // Memory region ID (-1 for null array)
    uint64_t byte_offset;  // Byte offset within region
    int64_t shape[APIC_MAX_DIMS];  // Array shape
    int64_t strides[APIC_MAX_DIMS];  // Array strides
    uint32_t element_size;  // Element size in bytes
    uint32_t _pad;
} APICArrayBindingRecord;  // 88 bytes

// Scalar parameter binding (fixed max size)
typedef struct {
    uint8_t type;  // APIC_PARAM_SCALAR
    uint8_t _pad1;
    uint16_t param_index;  // Parameter index in kernel signature
    uint16_t size;  // Actual size of scalar value
    uint16_t _pad2;
    uint8_t value[APIC_MAX_SCALAR_SIZE];  // Scalar value bytes
} APICScalarBindingRecord;  // 136 bytes

// -----------------------------------------------------------------------------
// Memory Operations
// -----------------------------------------------------------------------------

// Memcpy Host-to-Device (variable: has inline data)
typedef struct {
    APICOpHeader header;  // op_type = APIC_OP_MEMCPY_H2D
    int32_t dst_region_id;
    uint32_t _pad;
    uint64_t dst_offset;
    uint64_t size;
    // uint8_t data[size] follows
} APICMemcpyH2DRecord;  // 32 bytes fixed

// Memcpy Device-to-Device (fixed size)
typedef struct {
    APICOpHeader header;  // op_type = APIC_OP_MEMCPY_D2D
    int32_t dst_region_id;
    int32_t src_region_id;
    uint64_t dst_offset;
    uint64_t src_offset;
    uint64_t size;
} APICMemcpyD2DRecord;  // 40 bytes

// Memset (fixed size)
typedef struct {
    APICOpHeader header;  // op_type = APIC_OP_MEMSET
    int32_t region_id;
    int32_t value;
    uint64_t offset;
    uint64_t size;
} APICMemsetRecord;  // 32 bytes

// In-graph allocation (fixed size)
typedef struct {
    APICOpHeader header;  // op_type = APIC_OP_ALLOC
    int32_t region_id;
    uint32_t _pad;
    uint64_t size;
} APICAllocRecord;  // 24 bytes

// =============================================================================
// Memory Section Records
// =============================================================================

// Memory region record
typedef struct {
    uint32_t region_id;
    uint32_t element_size;
    uint64_t size;  // Size in bytes
    uint8_t role;  // APICMemoryRole
    uint8_t has_initial_data;  // 1 if initial_data follows
    uint8_t _pad[6];
    // If has_initial_data: uint8_t initial_data[size] follows
} APICMemoryRegionRecord;  // 24 bytes fixed

// =============================================================================
// Recording API Structures (for passing info from Python to C++)
// =============================================================================

// Parameter binding info for recording (passed from Python during kernel launch)
typedef struct {
    uint8_t type;  // APIC_PARAM_ARRAY or APIC_PARAM_SCALAR
    uint8_t ndim;  // For arrays: number of dimensions
    uint16_t param_index;
    int32_t region_id;  // For arrays: memory region ID (-1 for null)
    uint64_t byte_offset;  // For arrays: byte offset within region
    int64_t shape[APIC_MAX_DIMS];
    int64_t strides[APIC_MAX_DIMS];
    uint32_t element_size;  // For arrays: element size in bytes
    uint16_t scalar_size;  // For scalars: size of scalar value
    uint8_t _pad[2];
    uint8_t scalar_value[APIC_MAX_SCALAR_SIZE];  // For scalars: raw value bytes
} APICParamBindingInfo;  // 216 bytes

// Launch info passed to wp_cuda_launch_kernel() for APIC recording
// Only includes fields needed to identify the kernel - other launch parameters
// (dim, block_dim, smem_bytes) are passed directly to wp_cuda_launch_kernel(),
// and shape/ndim are in launch_bounds_t which is always args[0].
typedef struct {
    const char* kernel_key;  // Kernel identifier string
    const char* module_hash;  // Module hash string
    uint8_t is_forward;  // 1 for forward, 0 for backward
    uint8_t _pad[3];
    const APICParamBindingInfo* params;  // Array of parameter bindings
    int32_t num_params;  // Number of parameter bindings
} APICLaunchInfo;

#pragma pack(pop)

// =============================================================================
// Execution Structures (must match runtime types in builtin.h and array.h)
// =============================================================================

// Launch bounds - must match layout of launch_bounds_t in builtin.h
typedef struct {
    int shape[APIC_LAUNCH_MAX_DIMS];
    int ndim;
    size_t size;
} apic_launch_bounds_t;

// Array descriptor - must match layout of array_t<T> in array.h
// Note: Uses uint64_t for pointers to be C-compatible
typedef struct {
    uint64_t data;  // Device pointer
    uint64_t grad;  // Gradient pointer (usually 0)
    int shape[APIC_MAX_DIMS];
    int strides[APIC_MAX_DIMS];
    int ndim;
} apic_array_t;

#ifdef __cplusplus
}
#endif
