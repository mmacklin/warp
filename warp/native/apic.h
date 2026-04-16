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
typedef struct APICStateInternal* APICState;

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

// Check if any APIC recording is active on the current thread.
// Callable from any compilation unit (wraps access to thread-local g_apic_state).
WP_API int wp_apic_is_recording_active(void);

// Record a host-to-host memcpy to the active APIC state.
// Called from warp.cpp host memory function hooks.
WP_API void wp_apic_record_host_memcpy(void* dst, void* src, size_t size);

// Record a host memset to the active APIC state.
// Called from warp.cpp host memory function hooks.
WP_API void wp_apic_record_host_memset(void* dst, int value, size_t size);

// Record a host memtile (repeat pattern) to the active APIC state.
// Must be called AFTER execution so the full dst content can be captured as inline data.
WP_API void wp_apic_record_host_memtile(void* dst, size_t total_size);

// Record a CPU kernel launch to the active APIC state.
// Called from Python before the kernel is invoked via ctypes (which handles
// the by-value launch_bounds_t<N> calling convention correctly).
// bounds: pointer to launch_bounds_t<N>; ndim: N (so C++ knows the layout).
WP_API void wp_apic_record_cpu_launch(
    void* kernel_fn,
    void* bounds,
    int ndim,
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

#ifdef __cplusplus
}  // extern "C"
#endif

// =============================================================================
// APIC Internal API (C++ only)
// =============================================================================

#ifdef __cplusplus

// Thread-local APIC state (set during recording)
extern thread_local APICState g_apic_state;

// Helper to check if APIC is recording (hides struct internals)
bool apic_is_recording(APICState state);

// Internal recording functions (called from wp_cuda_launch_kernel, wp_memcpy_*, etc.)
void apic_record_kernel_launch(
    APICStateInternal* state,
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
    int num_params
);

void apic_record_memcpy(APICStateInternal* state, void* dst, void* src, size_t size, APICOpType kind);

void apic_record_memset(APICStateInternal* state, void* dst, int value, size_t size);

void apic_record_alloc(APICStateInternal* state, void* ptr, size_t size);

#endif  // __cplusplus
