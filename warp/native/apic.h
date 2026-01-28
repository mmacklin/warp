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
// POD structs for serialization are in apic_types.h

#include "apic_types.h"

// Forward declaration of internal state (definition in warp.cu)
struct APICStateInternal;

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
    const APICParamBindingInfo* params,
    int num_params
);

void apic_record_memcpy(APICStateInternal* state, void* dst, void* src, size_t size, APICOpType kind);

void apic_record_memset(APICStateInternal* state, void* dst, int value, size_t size);

void apic_record_alloc(APICStateInternal* state, void* ptr, size_t size);
