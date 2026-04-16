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

#define APIC_FORMAT_VERSION 3
#define APIC_MAGIC "WGF1"
#define APIC_MAGIC_VALUE 0x31464757  // "WGF1" as little-endian uint32

// Maximum array dimensions (matches Warp's ARRAY_MAX_DIMS)
#define APIC_MAX_DIMS 4


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
    APIC_OP_MEMCPY_H2H = 7,  // Host-to-host memcpy (CPU graphs, uses APICMemcpyD2DRecord format)
    APIC_OP_ARRAY_COPY = 8,  // Strided array copy (non-contiguous wp.copy)

    // Future: high-level Warp operations
    // APIC_OP_MESH_CREATE = 10,
    // APIC_OP_VOLUME_CREATE = 11,
    // APIC_OP_BVH_CREATE = 12,
} APICOpType;

// Device type for graph target
#define APIC_DEVICE_CUDA 0
#define APIC_DEVICE_CPU  1


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
// Mesh Serialization Records
// =============================================================================

// Mesh data record - stores all info needed to reconstruct a mesh
typedef struct {
    int32_t num_points;
    int32_t num_tris;
    uint8_t support_winding_number;
    uint8_t bvh_constructor;
    uint16_t bvh_leaf_size;
    uint32_t points_region_id;
    uint32_t indices_region_id;
    uint32_t velocities_region_id;  // UINT32_MAX if absent
    uint64_t original_ptr;
} APICMeshRecord;  // 32 bytes

// Handle pointer location in a memory region (for fixup) - works for Mesh, Volume, BVH, etc.
typedef struct {
    uint32_t region_id;
    uint32_t _pad;
    uint64_t offset;
    uint64_t stride;  // 0 = single pointer
} APICPtrLocationRecord;  // 24 bytes

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

    // Launch bounds (embedded, was previously param[0])
    int32_t shape[APIC_LAUNCH_MAX_DIMS];  // Launch shape
    int32_t ndim;  // Number of dimensions
    uint64_t size;  // Total threads (same as dim below, for launch_bounds_t compatibility)

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
    uint16_t num_params;  // Number of parameter bindings (array params only, starting from index 1)
    uint16_t num_handle_offsets;  // Number of handle byte offsets

    // Variable data follows in order:
    // 1. char kernel_key[kernel_key_len]
    // 2. char module_hash[module_hash_len]
    // 3. Parameter bindings (APICLaunchParamRecord for each array param)
    // 4. uint32_t handle_offsets[num_handle_offsets] - byte offsets where handles are in params buffer
} APICLaunchRecord;

// Parameter binding record for array or scalar parameters (param_index >= 1)
// For arrays: uses region_id, byte_offset, shape, strides, element_size
// For scalars: is_array=0, scalar_size in byte_offset, value bytes in shape[] and strides[]
typedef struct {
    uint8_t is_array;  // 1 for array, 0 for scalar
    uint8_t ndim;  // Number of dimensions (arrays only)
    uint16_t param_index;  // Parameter index in kernel signature (1-based, 0 is launch_bounds)
    int32_t region_id;  // Memory region ID (-1 for null array or scalar)
    uint64_t byte_offset;  // Byte offset within region (arrays) or scalar_size (scalars)
    int64_t shape[APIC_MAX_DIMS];  // Array shape or first 32 bytes of scalar value
    int64_t strides[APIC_MAX_DIMS];  // Array strides or next 32 bytes of scalar value
    uint32_t element_size;  // Element size in bytes (arrays only)
    uint32_t _pad1;
} APICLaunchParamRecord;  // 88 bytes

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

// Strided array copy (for non-contiguous wp.copy)
// Stores full array descriptors so the copy can be replayed with correct strides
typedef struct {
    APICOpHeader header;  // op_type = APIC_OP_ARRAY_COPY
    // Destination
    int32_t dst_region_id;
    int32_t dst_type;  // ARRAY_TYPE_*
    uint64_t dst_offset;  // byte offset of data within region
    int32_t dst_shape[APIC_MAX_DIMS];
    int32_t dst_strides[APIC_MAX_DIMS];
    int32_t dst_ndim;
    // Source
    int32_t src_region_id;
    int32_t src_type;
    uint64_t src_offset;
    int32_t src_shape[APIC_MAX_DIMS];
    int32_t src_strides[APIC_MAX_DIMS];
    int32_t src_ndim;
    // Common
    int32_t elem_size;
} APICArrayCopyRecord;

// =============================================================================
// Memory Section Records
// =============================================================================

// Memory region record
typedef struct {
    uint32_t region_id;
    uint32_t element_size;
    uint64_t size;  // Size in bytes
    uint8_t has_initial_data;  // 1 if initial_data follows
    uint8_t _pad[7];
    // If has_initial_data: uint8_t initial_data[size] follows
} APICMemoryRegionRecord;  // 24 bytes fixed

// =============================================================================
// Recording API Structures (for passing info from Python to C++)
// =============================================================================

// APICLaunchParam is the same as APICLaunchParamRecord
typedef APICLaunchParamRecord APICLaunchParam;

// Launch info passed to wp_cuda_launch_kernel() for APIC recording
// Only includes fields needed to identify the kernel - other launch parameters
// (dim, block_dim, smem_bytes) are passed directly to wp_cuda_launch_kernel(),
// and shape/ndim are in launch_bounds_t which is always args[0].
typedef struct {
    const char* kernel_key;  // Kernel identifier string
    const char* module_hash;  // Module hash string
    uint8_t is_forward;  // 1 for forward, 0 for backward
    uint8_t ndim;  // Number of launch dimensions (1-4), for parsing launch_bounds_t<N>
    uint8_t _pad[2];
    const APICLaunchParam* params;  // Array of parameter bindings
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
