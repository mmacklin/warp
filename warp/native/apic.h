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
// This header contains internal definitions used by the APIC implementation.
// Public API declarations are in warp.h

#include <cstddef>
#include <cstdint>

// Operation types
enum APICOpType : uint8_t {
    APIC_OP_KERNEL_LAUNCH = 1,
    APIC_OP_MEMCPY_H2D = 2,
    APIC_OP_MEMCPY_D2H = 3,
    APIC_OP_MEMCPY_D2D = 4,
    APIC_OP_MEMSET = 5,
    APIC_OP_ALLOC = 6,
};

// Parameter types for kernel launches
enum APICParamType : uint8_t {
    APIC_PARAM_ARRAY = 1,
    APIC_PARAM_SCALAR = 2,
};

// Maximum dimensions for arrays (matches Warp's ARRAY_MAX_DIMS)
#define APIC_MAX_DIMS 4

// Memory region roles (must match warp.h APICMemoryRole enum)
enum APICMemoryRoleInternal : uint8_t {
    APIC_ROLE_INTERNAL_INT = 0,
    APIC_ROLE_INPUT_INT = 1,
    APIC_ROLE_OUTPUT_INT = 2,
    APIC_ROLE_INPUT_OUTPUT_INT = 3,
};

// Memory region structure for internal use
struct APICMemoryRegion {
    uint32_t region_id;
    uint64_t base_ptr;
    uint64_t size;
    uint32_t element_size;
    uint8_t role;  // APICMemoryRole
};

// Forward declaration of internal state (definition in warp.cu)
struct APICStateInternal;

// Internal recording functions (called from wp_cuda_launch_kernel, wp_memcpy_*, etc.)
void apic_record_kernel_launch(
    APICStateInternal* state,
    void* kernel,
    size_t dim,
    int max_blocks,
    int block_dim,
    int smem_bytes,
    void** args,
    size_t num_args,
    size_t* arg_sizes
);

void apic_record_memcpy(APICStateInternal* state, void* dst, void* src, size_t size, APICOpType kind);

void apic_record_memset(APICStateInternal* state, void* dst, int value, size_t size);

void apic_record_alloc(APICStateInternal* state, void* ptr, size_t size);
