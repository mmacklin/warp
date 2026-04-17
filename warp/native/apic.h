/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// APIC (API Capture) - Records Warp API calls for serialization and replay
// This header contains both public API and internal definitions.
// POD structs for serialization are in apic_types.h

#include "apic_types.h"

#include <stddef.h>
#include <stdint.h>

#ifndef WP_API
#ifdef _WIN32
#define WP_API __declspec(dllexport)
#else
#define WP_API __attribute__((visibility("default")))
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// APIC Public API - Recording and Saving
// =============================================================================

// Opaque handle to APIC state (used internally during CUDA graph capture)
typedef struct APICGraphInternal* APICState;

// APIC State Management
WP_API APICState wp_apic_create_state();
WP_API void wp_apic_destroy_state(APICState state);

// Recording Control
WP_API void wp_apic_begin_recording(APICState state);
WP_API void wp_apic_end_recording(APICState state);
WP_API int wp_apic_is_recording(APICState state);

// State Queries
WP_API uint32_t wp_apic_get_operation_count(APICState state);
WP_API uint32_t wp_apic_get_memory_region_count(APICState state);
WP_API uint32_t wp_apic_get_module_count(APICState state);
WP_API uint32_t wp_apic_get_kernel_count(APICState state);

// Memory Region Registration
WP_API uint32_t
wp_apic_register_memory_region(APICState state, uint64_t base_ptr, uint64_t size, uint32_t element_size);

// Metadata Registration - call these before wp_apic_state_save()
WP_API void wp_apic_register_module(
    APICState state, const char* module_hash, const char* module_name, const char* cubin_filename, int target_arch
);

WP_API void wp_apic_register_kernel(
    APICState state,
    const char* kernel_key,
    const char* module_hash,
    const char* forward_name,
    const char* backward_name,  // can be NULL or empty string
    int forward_smem_bytes,
    int backward_smem_bytes,
    int block_dim
);

WP_API void wp_apic_register_binding(APICState state, const char* name, uint32_t region_id);

// Register handle pointer location within a memory region (for fixup during replay)
// Called automatically during track_array() when dtype contains handles
// stride=0 means single pointer, otherwise stride is the array element size
WP_API void wp_apic_register_ptr_location(APICState state, uint32_t region_id, uint64_t offset, uint64_t stride);

// Save APIC state to a WGF file
// Serializes metadata from registered modules/kernels/params
// Returns 1 on success, 0 on failure
WP_API int wp_apic_state_save(APICState state, const char* path, uint32_t target_arch);

// =============================================================================
// APIC Public API - Loading and Execution
// =============================================================================

// Opaque handle to a loaded APIC graph
typedef struct APICGraphInternal* APICGraph;

// Load a graph from a .wgf file
// Returns NULL on failure (use wp_get_error_string() for details)
WP_API APICGraph wp_apic_load_graph(void* context, const char* path);

// Destroy a loaded graph and free all resources
WP_API void wp_apic_destroy_graph(APICGraph graph);

// Set a named parameter by copying data to the pre-allocated region
// This copies host data to the device memory region associated with the parameter
// Returns 1 on success, 0 on failure (name not found or size mismatch)
WP_API int wp_apic_set_param(APICGraph graph, const char* name, const void* data, size_t size);

// Get a named parameter by copying data from the pre-allocated region
// This copies device data from the memory region to the destination pointer
// Returns 1 on success, 0 on failure (name not found or size mismatch)
WP_API int wp_apic_get_param(APICGraph graph, const char* name, void* data, size_t size);

// Get a named parameter's device pointer (for direct access)
// Returns the device pointer, or NULL if not found
WP_API void* wp_apic_get_param_ptr(APICGraph graph, const char* name);

// Get the CUDA graph handle
// Returns the cudaGraph_t handle, or NULL on failure
WP_API void* wp_apic_get_cuda_graph(APICGraph graph);

// Get the instantiated CUDA graph executable (creates if needed)
// Returns the cudaGraphExec_t handle, or NULL on failure
// Users should call wp_cuda_graph_launch() with the returned exec to launch
WP_API void* wp_apic_get_cuda_graph_exec(APICGraph graph);

// Query functions
WP_API int wp_apic_get_num_params(APICGraph graph);
WP_API const char* wp_apic_get_param_name(APICGraph graph, int index);
WP_API size_t wp_apic_get_param_size(APICGraph graph, const char* name);

// =============================================================================
// APIC Public API - CPU Graph Support
// =============================================================================

// Record a memtile (repeat small pattern N times) to the active APIC state.
// Shared by both CPU and CUDA — stores the pattern inline for replay.
// src points to the fill pattern (srcsize bytes, may be host memory).
WP_API void wp_apic_record_memtile(void* dst, const void* src, size_t srcsize, size_t n);

// Record a strided array copy (non-contiguous wp.copy) to the active APIC state.
// dst/src are pointers to array_t<void> descriptors. Only ARRAY_TYPE_REGULAR supported.
WP_API void wp_apic_record_array_copy(void* dst, void* src, int dst_type, int src_type, int elem_size);

// Launch a host kernel with optional APIC recording.
// Records the launch if APIC capture is active, then executes the kernel.
// Mirrors wp_cuda_launch_kernel() — Python calls this one function, recording
// is handled internally, invisible to the caller.
// bounds: pointer to launch_bounds_t<N>; ndim: N for by-value dispatch;
// args/adj_args: packed kernel arg structs; apic_info: NULL when not capturing.
WP_API void wp_launch_host_kernel(
    void* kernel_fn,
    void* bounds,
    int ndim,
    void* args,
    void* adj_args,
    const APICLaunchInfo* apic_info
);

// Register a host function pointer for CPU graph replay.
// Must be called during capture for each unique kernel.
WP_API void wp_apic_register_host_function(
    APICState state,
    const char* kernel_key,
    void* forward_fn,
    void* backward_fn
);

// Replay all operations in the state's operation stream on CPU.
// Uses registered host functions for kernel launches, memcpy/memset for memory ops.
// Returns 1 on success, 0 on failure.
WP_API int wp_apic_replay_host_ops(APICState state);

// Mark APIC state as targeting CPU (affects serialization behavior).
WP_API void wp_apic_set_cpu_mode(APICState state);

// Register host function on a loaded graph (for CPU graph replay after loading).
// Python calls this after loading .o modules via wp_load_obj + wp_lookup to provide
// the function pointers that wp_apic_replay_loaded_host_graph() needs.
WP_API void wp_apic_graph_register_host_function(APICGraph graph, const char* kernel_key, void* forward_fn, void* backward_fn);

// Replay a loaded CPU graph using registered host functions.
// Operates on an APICGraphInternal loaded via wp_apic_load_graph(NULL, path).
// Returns 1 on success, 0 on failure.
WP_API int wp_apic_replay_loaded_host_graph(APICGraph graph);

#ifdef __cplusplus
}  // extern "C"
#endif

// =============================================================================
// APIC Internal API (C++ only)
// =============================================================================

#ifdef __cplusplus

#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Forward declarations for CUDA types used inside APICGraphInternal.
// When WP_ENABLE_CUDA is 0 (or undefined), these are not needed because
// the CUDA-specific fields are compiled out.
#if WP_ENABLE_CUDA
#include "cuda_util.h"  // CUgraph, CUgraphExec, CUmodule, ContextGuard, etc.
#endif

// ============================================================================
// APIC Internal Structures (shared between apic.cpp and apic.cu)
// ============================================================================

// Module info (used for both recording and loaded state)
struct APICModule {
    std::string module_hash;
    std::string module_name;
    std::string cubin_filename;
    int target_arch;
#if WP_ENABLE_CUDA
    CUmodule cuda_module = nullptr;  // Set after loading
#endif
};

// Kernel info (used for both recording and loaded state)
struct APICKernel {
    std::string kernel_key;
    std::string module_hash;
    std::string forward_name;
    std::string backward_name;
    int forward_smem_bytes;
    int backward_smem_bytes;
    int block_dim;
};

// Memory region info (used for both recording and loaded state)
struct APICRegion {
    uint32_t region_id = 0;
    uint64_t base_ptr = 0;  // Original device pointer during recording
    uint64_t size = 0;
    uint32_t element_size = 0;
    std::vector<uint8_t> initial_data;  // For internal regions during recording
    void* ptr = nullptr;  // Allocated device pointer after loading
};

// Handle pointer location in a memory region (for fixup during replay)
struct APICPtrLocation {
    uint32_t region_id;
    uint64_t offset;
    uint64_t stride;  // 0 = single pointer
};

// Merged internal structure for both recording (APICState) and loaded graph (APICGraph).
// Some fields are only used during recording, some only during loading -- unused fields
// stay at their default values.
struct APICGraphInternal {
    // === Recording state (populated during capture) ===
    bool recording = false;
    std::unordered_map<uint64_t, APICRegion> memory_regions;  // keyed by base_ptr (recording)
    uint32_t next_region_id = 0;

    // === Loaded graph state (populated during file loading) ===
    void* cuda_context = nullptr;
    int target_arch = 0;
    std::unordered_map<uint32_t, APICRegion> regions;  // keyed by region_id (loaded)
    std::unordered_map<std::string, uint32_t> bindings;  // name -> region_id (loaded)
    std::vector<std::string> binding_names;  // ordered for indexing (loaded)
    std::unordered_map<uint64_t, uint64_t> handle_ptr_remap;
    std::vector<APICMeshRecord> mesh_records;
    std::vector<uint64_t> created_mesh_ids;
#if WP_ENABLE_CUDA
    CUgraph cuda_graph = nullptr;
    CUgraphExec cuda_graph_exec = nullptr;
#endif
    std::string base_path;

    // === Shared state (used by both recording and loaded paths) ===
    std::vector<uint8_t> operation_stream;
    uint32_t operation_count = 0;
    std::unordered_map<std::string, APICModule> modules;
    std::unordered_map<std::string, APICKernel> kernels;
    std::vector<std::pair<std::string, uint32_t>> recording_bindings;  // name->id pairs (recording)
    std::vector<APICPtrLocation> ptr_locations;
    bool is_cpu = false;
    std::unordered_map<std::string, std::pair<void*, void*>> host_functions;
    std::unordered_map<uint32_t, uint64_t> region_id_to_ptr;  // region_id -> base_ptr (both)

    // === Methods ===

    void* resolve_region_ptr(int32_t region_id, uint64_t offset) const
    {
        auto it = region_id_to_ptr.find(region_id);
        if (it != region_id_to_ptr.end())
            return (void*)(it->second + offset);
        return nullptr;
    }

    void append_bytes(const void* data, size_t size)
    {
        size_t off = operation_stream.size();
        operation_stream.resize(off + size);
        memcpy(operation_stream.data() + off, data, size);
    }

    bool find_region(uint64_t ptr, int32_t& region_id, uint64_t& offset) const
    {
        for (const auto& kv : memory_regions) {
            const APICRegion& r = kv.second;
            if (ptr >= r.base_ptr && ptr < r.base_ptr + r.size) {
                region_id = r.region_id;
                offset = ptr - r.base_ptr;
                return true;
            }
        }
        return false;
    }

    // Destructor handles both recording and loaded cleanup.
    // Defined out-of-line in apic.cpp because it references symbols
    // (wp_mesh_destroy_device, cudaFree, etc.) not yet declared at this point.
    ~APICGraphInternal();
};

// ============================================================================
// APIC Internal Function Declarations
// ============================================================================

// Helper: extract shape and size from a launch_bounds_t<N> given N.
// Accounts for alignment padding between shape[N] and size_t size.
void apic_parse_launch_bounds(const void* bounds, int ndim, int* out_shape, size_t* out_size);

// Thread-local APIC state (set during recording)
extern thread_local APICState g_apic_state;

// Helper to check if APIC is recording (hides struct internals)
bool apic_is_recording(APICState state);

// Internal recording functions (called from wp_cuda_launch_kernel, wp_memcpy_*, etc.)
// scalar_data: optional trailing buffer for scalar params larger than 64 bytes
//   (each such param's shape[0] holds the byte offset into scalar_data).
void apic_record_kernel_launch(
    APICGraphInternal* state,
    void* kernel,
    size_t dim,
    const int* shape,
    int ndim,
    int max_blocks,
    int block_dim,
    int smem_bytes,
    bool is_forward,
    const char* kernel_key,
    const char* module_hash,
    const APICLaunchParam* params,
    int num_params,
    const void* scalar_data,
    uint32_t scalar_data_size
);

void apic_record_memcpy(APICGraphInternal* state, void* dst, void* src, size_t size, APICOpType kind);

void apic_record_memset(APICGraphInternal* state, void* dst, int value, size_t size);

void apic_record_alloc(APICGraphInternal* state, void* ptr, size_t size);

#if WP_ENABLE_CUDA
// CUDA-only functions (defined in apic.cu, called from apic.cpp).
// extern "C" for stable linkage across translation units.
extern "C" {
WP_API CUfunction apic_get_kernel_function(
    APICGraphInternal* graph,
    const char* module_hash,
    size_t hash_len,
    const char* kernel_key,
    size_t key_len,
    int is_forward
);

WP_API bool apic_rebuild_cuda_graph(APICGraphInternal* graph, CUstream stream);

WP_API bool apic_create_meshes(APICGraphInternal* graph);
}  // extern "C"
#endif  // WP_ENABLE_CUDA

#endif  // __cplusplus
