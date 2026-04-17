/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

// CUDA-only APIC (API Capture) functions.
// This file is included at the end of warp.cu.
// Platform-independent code lives in apic.cpp.

#include "mesh.h"  // For wp::Mesh, wp_mesh_create_device

#include <memory>  // For std::make_unique, std::unique_ptr

// Helper: get kernel function (looks up or retrieves from cache)
extern "C" WP_API CUfunction apic_get_kernel_function(
    APICGraphInternal* graph,
    const char* module_hash,
    size_t hash_len,
    const char* kernel_key,
    size_t key_len,
    int is_forward
)
{
    std::string hash_str(module_hash, hash_len);
    std::string key_str(kernel_key, key_len);

    auto mod_it = graph->modules.find(hash_str);
    if (mod_it == graph->modules.end() || !mod_it->second.cuda_module) {
        wp::set_error_string("Module not loaded: %s", hash_str.c_str());
        return nullptr;
    }

    // Get kernel name from metadata
    auto kern_it = graph->kernels.find(key_str);
    if (kern_it == graph->kernels.end()) {
        wp::set_error_string("Kernel not found: %s", key_str.c_str());
        return nullptr;
    }

    const std::string& kernel_name = is_forward ? kern_it->second.forward_name : kern_it->second.backward_name;

    CUfunction kernel;
    CUresult err = cuModuleGetFunction_f(&kernel, mod_it->second.cuda_module, kernel_name.c_str());
    if (err != CUDA_SUCCESS) {
        wp::set_error_string("Failed to get kernel function %s: %d", kernel_name.c_str(), err);
        return nullptr;
    }

    return kernel;
}

// Rebuild CUDA graph by replaying operations from the stream
extern "C" WP_API bool apic_rebuild_cuda_graph(APICGraphInternal* graph, CUstream stream)
{
    // Destroy old graph using runtime API
    if (graph->cuda_graph_exec) {
        cudaGraphExecDestroy((cudaGraphExec_t)graph->cuda_graph_exec);
        graph->cuda_graph_exec = nullptr;
    }
    if (graph->cuda_graph) {
        cudaGraphDestroy((cudaGraph_t)graph->cuda_graph);
        graph->cuda_graph = nullptr;
    }

    // Begin capture using runtime API
    cudaError_t cuda_err = cudaStreamBeginCapture((cudaStream_t)stream, cudaStreamCaptureModeThreadLocal);
    if (cuda_err != cudaSuccess) {
        wp::set_error_string("Failed to begin graph capture: %d", cuda_err);
        return false;
    }

    bool success = true;
    CUresult err;

    // Iterate through operation stream
    const uint8_t* ptr = graph->operation_stream.data();
    const uint8_t* end = ptr + graph->operation_stream.size();

    for (uint32_t i = 0; i < graph->operation_count && ptr < end && success; i++) {
        const APICOpHeader* header = reinterpret_cast<const APICOpHeader*>(ptr);
        const uint8_t* op_start = ptr;

        switch (header->op_type) {
        case APIC_OP_KERNEL_LAUNCH: {
            const APICLaunchRecord* rec = reinterpret_cast<const APICLaunchRecord*>(ptr);
            const uint8_t* var_data = ptr + sizeof(APICLaunchRecord);

            // Parse strings: kernel_key followed by module_hash
            const char* kernel_key = reinterpret_cast<const char*>(var_data);
            const char* module_hash = reinterpret_cast<const char*>(var_data + rec->kernel_key_len);

            // Get kernel function
            CUfunction kernel = apic_get_kernel_function(
                graph, module_hash, rec->module_hash_len, kernel_key, rec->kernel_key_len, rec->is_forward != 0
            );
            if (!kernel) {
                success = false;
                break;
            }

            // Skip past strings to param bindings
            const uint8_t* params_ptr = var_data + rec->kernel_key_len + rec->module_hash_len;

            // Trailing scalar overflow section (for scalars > 64 bytes).
            // Sits immediately after the parameter records.
            const uint8_t* scalar_data = params_ptr + rec->num_params * sizeof(APICLaunchParamRecord);
            size_t used_so_far = sizeof(APICLaunchRecord) + rec->kernel_key_len + rec->module_hash_len
                + rec->num_params * sizeof(APICLaunchParamRecord);
            size_t scalar_data_size
                = (rec->header.total_size > used_so_far) ? (rec->header.total_size - used_so_far) : 0;

            std::vector<void*> args;
            std::vector<std::unique_ptr<uint8_t[]>> arg_storage;

            // Create launch_bounds_t<N> as param[0] from embedded data in record.
            // Layout: int shape[N], size_t size, bool tiled (varies by N).
            int ndim = rec->ndim;
            size_t shape_bytes = ndim * sizeof(int);
            size_t size_offset = (shape_bytes + sizeof(size_t) - 1) & ~(sizeof(size_t) - 1);
            size_t bounds_total = size_offset + sizeof(size_t) + sizeof(bool);
            auto bounds = std::make_unique<uint8_t[]>(bounds_total);
            memset(bounds.get(), 0, bounds_total);
            int* shape_ptr = reinterpret_cast<int*>(bounds.get());
            for (int d = 0; d < ndim && d < APIC_LAUNCH_MAX_DIMS; d++)
                shape_ptr[d] = rec->shape[d];
            *reinterpret_cast<size_t*>(bounds.get() + size_offset) = rec->size;
            *reinterpret_cast<bool*>(bounds.get() + size_offset + sizeof(size_t)) = false;
            args.push_back(bounds.get());
            arg_storage.push_back(std::move(bounds));

            // Parse param bindings (arrays and scalars)
            for (uint16_t j = 0; j < rec->num_params; j++) {
                const APICLaunchParamRecord* binding = reinterpret_cast<const APICLaunchParamRecord*>(params_ptr);
                params_ptr += sizeof(APICLaunchParamRecord);

                if (binding->is_array) {
                    // Array parameter - create array_t structure
                    auto arr = std::make_unique<uint8_t[]>(sizeof(apic_array_t));
                    apic_array_t* arr_ptr = reinterpret_cast<apic_array_t*>(arr.get());
                    memset(arr_ptr, 0, sizeof(apic_array_t));

                    void* resolved = graph->resolve_region_ptr(binding->region_id, binding->byte_offset);
                    arr_ptr->data = (uint64_t)resolved;
                    arr_ptr->grad = 0;
                    arr_ptr->ndim = binding->ndim;
                    for (int d = 0; d < binding->ndim && d < APIC_MAX_DIMS; d++) {
                        arr_ptr->shape[d] = (int)binding->shape[d];
                        arr_ptr->strides[d] = (int)binding->strides[d];
                    }

                    args.push_back(arr_ptr);
                    arg_storage.push_back(std::move(arr));
                } else {
                    // Scalar parameter: value bytes are stored inline in shape[]/strides[]
                    // for sizes <= 64 bytes, or in the trailing scalar_data section for
                    // larger scalars (shape[0] holds the offset into scalar_data).
                    constexpr size_t MAX_INLINE_SCALAR = APIC_MAX_DIMS * sizeof(int64_t) * 2;
                    size_t scalar_size = binding->byte_offset;
                    auto scalar = std::make_unique<uint8_t[]>(scalar_size);

                    if (scalar_size > MAX_INLINE_SCALAR) {
                        // Overflow scalar -- bytes are in trailing section.
                        uint64_t src_off = static_cast<uint64_t>(binding->shape[0]);
                        if (!scalar_data || src_off + scalar_size > scalar_data_size) {
                            success = false;
                            break;
                        }
                        memcpy(scalar.get(), scalar_data + src_off, scalar_size);
                    } else {
                        // Inline: shape[] (first 32B) + strides[] (next 32B)
                        const uint8_t* shape_bytes = reinterpret_cast<const uint8_t*>(binding->shape);
                        const uint8_t* strides_bytes = reinterpret_cast<const uint8_t*>(binding->strides);
                        size_t first_part = std::min(scalar_size, (size_t)(APIC_MAX_DIMS * sizeof(int64_t)));
                        memcpy(scalar.get(), shape_bytes, first_part);
                        if (scalar_size > first_part) {
                            memcpy(scalar.get() + first_part, strides_bytes, scalar_size - first_part);
                        }
                    }

                    args.push_back(scalar.get());
                    arg_storage.push_back(std::move(scalar));
                }
            }

            // Calculate grid dimensions
            int num_threads = (int)rec->dim;
            int block_size = rec->block_dim;
            int max_blocks = rec->max_blocks;
            int num_blocks = (num_threads + block_size - 1) / block_size;
            if (max_blocks > 0 && num_blocks > max_blocks) {
                num_blocks = max_blocks;
            }

            // Launch kernel
            err = cuLaunchKernel_f(
                kernel, num_blocks, 1, 1, block_size, 1, 1, rec->smem_bytes, stream, args.data(), nullptr
            );

            if (err != CUDA_SUCCESS) {
                wp::set_error_string("Failed to launch kernel: %d", err);
                success = false;
            }
            break;
        }

        case APIC_OP_MEMCPY_H2D: {
            const APICMemcpyH2DRecord* rec = reinterpret_cast<const APICMemcpyH2DRecord*>(ptr);
            const uint8_t* src_data = ptr + sizeof(APICMemcpyH2DRecord);
            void* dst_ptr = graph->resolve_region_ptr(rec->dst_region_id, rec->dst_offset);
            cuda_err = cudaMemcpyAsync(dst_ptr, src_data, rec->size, cudaMemcpyHostToDevice, (cudaStream_t)stream);
            if (cuda_err != cudaSuccess) {
                wp::set_error_string("Failed H2D memcpy: %d", cuda_err);
                success = false;
            }
            break;
        }

        case APIC_OP_MEMCPY_D2D: {
            const APICMemcpyD2DRecord* rec = reinterpret_cast<const APICMemcpyD2DRecord*>(ptr);
            void* dst_ptr = graph->resolve_region_ptr(rec->dst_region_id, rec->dst_offset);
            void* src_ptr = graph->resolve_region_ptr(rec->src_region_id, rec->src_offset);
            cuda_err = cudaMemcpyAsync(dst_ptr, src_ptr, rec->size, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
            if (cuda_err != cudaSuccess) {
                wp::set_error_string("Failed D2D memcpy: %d", cuda_err);
                success = false;
            }
            break;
        }

        case APIC_OP_MEMSET: {
            const APICMemsetRecord* rec = reinterpret_cast<const APICMemsetRecord*>(ptr);
            void* dst_ptr = graph->resolve_region_ptr(rec->region_id, rec->offset);
            cuda_err = cudaMemsetAsync(dst_ptr, rec->value, rec->size, (cudaStream_t)stream);
            if (cuda_err != cudaSuccess) {
                wp::set_error_string("Failed memset: %d", cuda_err);
                success = false;
            }
            break;
        }

        case APIC_OP_ARRAY_COPY: {
            const APICArrayCopyRecord* rec = reinterpret_cast<const APICArrayCopyRecord*>(ptr);
            void* dst_data = graph->resolve_region_ptr(rec->dst_region_id, rec->dst_offset);
            void* src_data = graph->resolve_region_ptr(rec->src_region_id, rec->src_offset);
            if (dst_data && src_data) {
                wp::array_t<void> dst_arr = {};
                dst_arr.data = dst_data;
                dst_arr.ndim = rec->dst_ndim;
                for (int d = 0; d < APIC_MAX_DIMS; d++) {
                    dst_arr.shape.dims[d] = rec->dst_shape[d];
                    dst_arr.strides[d] = rec->dst_strides[d];
                }
                wp::array_t<void> src_arr = {};
                src_arr.data = src_data;
                src_arr.ndim = rec->src_ndim;
                for (int d = 0; d < APIC_MAX_DIMS; d++) {
                    src_arr.shape.dims[d] = rec->src_shape[d];
                    src_arr.strides[d] = rec->src_strides[d];
                }
                wp_array_copy_device(
                    graph->cuda_context, &dst_arr, &src_arr, rec->dst_type, rec->src_type, rec->elem_size
                );
            }
            break;
        }

        case APIC_OP_MEMTILE: {
            // Replay memtile directly via cudaMemcpyAsync for each repetition.
            // We can't call wp_memtile_device here because it allocates temp memory
            // internally, which can deadlock during CUDA stream capture.
            const APICMemtileRecord* rec = reinterpret_cast<const APICMemtileRecord*>(ptr);
            void* dst = graph->resolve_region_ptr(rec->dst_region_id, rec->dst_offset);
            const void* pattern = ptr + sizeof(APICMemtileRecord);
            if (dst) {
                for (uint64_t i = 0; i < rec->n; i++) {
                    cuda_err = cudaMemcpyAsync(
                        (uint8_t*)dst + i * rec->srcsize, pattern, rec->srcsize, cudaMemcpyHostToDevice,
                        (cudaStream_t)stream
                    );
                    if (cuda_err != cudaSuccess) {
                        wp::set_error_string("Failed memtile H2D copy: %d", cuda_err);
                        success = false;
                        break;
                    }
                }
            }
            break;
        }

        case APIC_OP_ALLOC:
            // Allocations are handled by memory region setup, skip
            break;

        default:
            wp::set_error_string("Unknown operation type: %d", header->op_type);
            success = false;
            break;
        }

        // Advance to next operation
        ptr = op_start + header->total_size;
    }

    // End capture using runtime API
    cudaGraph_t captured_graph;
    cuda_err = cudaStreamEndCapture((cudaStream_t)stream, &captured_graph);
    if (cuda_err != cudaSuccess) {
        wp::set_error_string("Failed to end graph capture: %d", cuda_err);
        return false;
    }

    if (!success) {
        cudaGraphDestroy(captured_graph);
        return false;
    }

    graph->cuda_graph = (CUgraph)captured_graph;

    return true;
}

// Create meshes from stored mesh records after memory regions are allocated
// This populates handle_ptr_remap with old_mesh_ptr -> new_mesh_ptr mappings
extern "C" WP_API bool apic_create_meshes(APICGraphInternal* graph)
{
    for (const APICMeshRecord& rec : graph->mesh_records) {
        // Get region pointers
        auto points_it = graph->regions.find(rec.points_region_id);
        auto indices_it = graph->regions.find(rec.indices_region_id);

        if (points_it == graph->regions.end() || !points_it->second.ptr) {
            wp::set_error_string("Mesh points region %u not found", rec.points_region_id);
            return false;
        }
        if (indices_it == graph->regions.end() || !indices_it->second.ptr) {
            wp::set_error_string("Mesh indices region %u not found", rec.indices_region_id);
            return false;
        }

        // Build array_t for points
        wp::array_t<wp::vec3> points;
        points.data = (wp::vec3*)points_it->second.ptr;
        points.grad = nullptr;
        points.shape[0] = rec.num_points;
        points.strides[0] = sizeof(wp::vec3);
        points.ndim = 1;

        // Build array_t for indices
        wp::array_t<int> indices;
        indices.data = (int*)indices_it->second.ptr;
        indices.grad = nullptr;
        indices.shape[0] = rec.num_tris * 3;
        indices.strides[0] = sizeof(int);
        indices.ndim = 1;

        // Build array_t for velocities (optional)
        wp::array_t<wp::vec3> velocities = {};
        if (rec.velocities_region_id != UINT32_MAX) {
            auto vel_it = graph->regions.find(rec.velocities_region_id);
            if (vel_it != graph->regions.end() && vel_it->second.ptr) {
                velocities.data = (wp::vec3*)vel_it->second.ptr;
                velocities.grad = nullptr;
                velocities.shape[0] = rec.num_points;
                velocities.strides[0] = sizeof(wp::vec3);
                velocities.ndim = 1;
            }
        }

        // Create mesh
        uint64_t new_mesh_id = wp_mesh_create_device(
            graph->cuda_context, points, velocities, indices, rec.num_points, rec.num_tris, rec.support_winding_number,
            rec.bvh_constructor, nullptr,  // groups
            rec.bvh_leaf_size
        );

        if (new_mesh_id == 0) {
            wp::set_error_string("Failed to create mesh from serialized data");
            return false;
        }

        // Track mesh for cleanup
        graph->created_mesh_ids.push_back(new_mesh_id);

        // Add to handle remap table
        graph->handle_ptr_remap[rec.original_ptr] = new_mesh_id;
    }

    return true;
}

extern "C" WP_API void* wp_apic_get_cuda_graph(APICGraph graph)
{
    if (!graph)
        return nullptr;

    ContextGuard guard(graph->cuda_context);

    // Build graph once on first access
    if (!graph->cuda_graph) {
        CUstream stream;
        cuStreamCreate_f(&stream, CU_STREAM_DEFAULT);

        bool success = apic_rebuild_cuda_graph(graph, stream);

        cuStreamDestroy_f(stream);

        if (!success)
            return nullptr;
    }

    return graph->cuda_graph;
}

extern "C" WP_API void* wp_apic_get_cuda_graph_exec(APICGraph graph)
{
    if (!graph)
        return nullptr;

    ContextGuard guard(graph->cuda_context);

    // Ensure graph is up to date
    if (!wp_apic_get_cuda_graph(graph))
        return nullptr;

    // Instantiate if needed using runtime API
    if (!graph->cuda_graph_exec) {
        cudaError_t err = cudaGraphInstantiateWithFlags(
            (cudaGraphExec_t*)&graph->cuda_graph_exec, (cudaGraph_t)graph->cuda_graph, 0
        );
        if (err != cudaSuccess) {
            wp::set_error_string("Failed to instantiate graph: %d", err);
            return nullptr;
        }
    }

    return graph->cuda_graph_exec;
}
