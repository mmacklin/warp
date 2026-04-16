/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

// This file is included at the end of warp.cu
// It contains all APIC (API Capture) implementation code

#include <memory>  // For std::make_unique, std::unique_ptr

#include "mesh.h"  // For wp::Mesh struct

// ============================================================================
// APIC Internal Structures
// ============================================================================

// Module info (used for both recording and loaded state)
struct APICModule {
    std::string module_hash;
    std::string module_name;
    std::string cubin_filename;
    int target_arch;
    CUmodule cuda_module = nullptr;  // Set after loading
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
    uint32_t region_id;
    uint64_t base_ptr;  // Original device pointer during recording
    uint64_t size;
    uint32_t element_size;
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
    CUgraph cuda_graph = nullptr;
    CUgraphExec cuda_graph_exec = nullptr;
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

    // Destructor handles both recording and loaded cleanup
    ~APICGraphInternal()
    {
        // CUDA loaded graph cleanup
        if (cuda_context) {
            ContextGuard guard(cuda_context);
            for (uint64_t mesh_id : created_mesh_ids)
                wp_mesh_destroy_device(mesh_id);
            if (cuda_graph_exec)
                cudaGraphExecDestroy((cudaGraphExec_t)cuda_graph_exec);
            if (cuda_graph)
                cudaGraphDestroy((cudaGraph_t)cuda_graph);
            if (!is_cpu) {
                for (auto& pair : regions)
                    if (pair.second.ptr) cudaFree(pair.second.ptr);
            }
            for (auto& pair : modules)
                if (pair.second.cuda_module) cuModuleUnload_f(pair.second.cuda_module);
        }
        // CPU loaded graph cleanup (regions allocated with malloc)
        if (is_cpu) {
            for (auto& pair : regions)
                if (pair.second.ptr) free(pair.second.ptr);
        }
        // Recording state doesn't own any allocated memory (arrays belong to Python)
    }
};

// Thread-local APIC state (set during recording)
thread_local APICGraphInternal* g_apic_state = nullptr;

// Helper to check if APIC is recording (hides struct internals from warp.cu)
bool apic_is_recording(APICGraphInternal* state) { return state && state->recording; }

// ============================================================================
// APIC (API Capture) Implementation
// ============================================================================

// Access mesh descriptors registry from mesh.cpp for mesh serialization
namespace wp {
extern std::map<uint64_t, Mesh> g_mesh_descriptors;
}

APICState wp_apic_create_state() { return new APICGraphInternal(); }

void wp_apic_destroy_state(APICState state)
{
    if (state) {
        delete state;
    }
}

void wp_apic_begin_recording(APICState state)
{
    if (state) {
        state->recording = true;
        state->operation_stream.clear();
        state->operation_count = 0;
        state->memory_regions.clear();
        state->modules.clear();
        state->kernels.clear();
        state->recording_bindings.clear();
        state->ptr_locations.clear();
        state->next_region_id = 0;
        g_apic_state = state;
    }
}

void wp_apic_end_recording(APICState state)
{
    if (state) {
        state->recording = false;
        if (g_apic_state == state) {
            g_apic_state = nullptr;
        }
    }
}

int wp_apic_is_recording(APICState state)
{
    if (state) {
        return state->recording ? 1 : 0;
    }
    return 0;
}

uint32_t wp_apic_get_operation_count(APICState state)
{
    if (state) {
        return state->operation_count;
    }
    return 0;
}

uint32_t wp_apic_get_memory_region_count(APICState state)
{
    if (state) {
        return static_cast<uint32_t>(state->memory_regions.size());
    }
    return 0;
}

uint32_t wp_apic_get_module_count(APICState state)
{
    if (state) {
        return static_cast<uint32_t>(state->modules.size());
    }
    return 0;
}

uint32_t wp_apic_get_kernel_count(APICState state)
{
    if (state) {
        return static_cast<uint32_t>(state->kernels.size());
    }
    return 0;
}

uint32_t wp_apic_register_memory_region(APICState state, uint64_t base_ptr, uint64_t size, uint32_t element_size)
{
    if (!state)
        return UINT32_MAX;

    // Check if region already exists
    auto it = state->memory_regions.find(base_ptr);
    if (it != state->memory_regions.end()) {
        return it->second.region_id;
    }

    // Create new region
    uint32_t region_id = state->next_region_id++;
    APICRegion region;
    region.region_id = region_id;
    region.base_ptr = base_ptr;
    region.size = size;
    region.element_size = element_size;

    state->memory_regions[base_ptr] = region;
    state->region_id_to_ptr[region_id] = base_ptr;
    return region_id;
}

void wp_apic_register_module(
    APICState state, const char* module_hash, const char* module_name, const char* cubin_filename, int target_arch
)
{
    if (!state || !module_hash)
        return;

    std::string hash_str(module_hash);
    // Only register if not already present
    if (state->modules.find(hash_str) == state->modules.end()) {
        APICModule mod;
        mod.module_hash = hash_str;
        mod.module_name = module_name ? module_name : "";
        mod.cubin_filename = cubin_filename ? cubin_filename : "";
        mod.target_arch = target_arch;
        state->modules[hash_str] = mod;
    }
}

void wp_apic_register_kernel(
    APICState state,
    const char* kernel_key,
    const char* module_hash,
    const char* forward_name,
    const char* backward_name,
    int forward_smem_bytes,
    int backward_smem_bytes,
    int block_dim
)
{
    if (!state || !kernel_key)
        return;

    std::string key_str(kernel_key);
    // Only register if not already present
    if (state->kernels.find(key_str) == state->kernels.end()) {
        APICKernel kern;
        kern.kernel_key = key_str;
        kern.module_hash = module_hash ? module_hash : "";
        kern.forward_name = forward_name ? forward_name : "";
        kern.backward_name = backward_name ? backward_name : "";
        kern.forward_smem_bytes = forward_smem_bytes;
        kern.backward_smem_bytes = backward_smem_bytes;
        kern.block_dim = block_dim;
        state->kernels[key_str] = kern;
    }
}

void wp_apic_register_binding(APICState state, const char* name, uint32_t region_id)
{
    if (!state || !name)
        return;

    state->recording_bindings.push_back({ std::string(name), region_id });

    // Capture data for this param region if not already captured
    // This ensures input params have their data serialized
    for (auto& kv : state->memory_regions) {
        APICRegion& region = kv.second;
        if (region.region_id == region_id && region.initial_data.empty() && region.size > 0) {
            region.initial_data.resize(region.size);
            if (state->is_cpu) {
                // CPU: direct host memory copy
                memcpy(region.initial_data.data(), (void*)region.base_ptr, region.size);
            } else {
                // CUDA: device-to-host copy
                cudaError_t err = cudaMemcpy(
                    region.initial_data.data(), (void*)region.base_ptr, region.size, cudaMemcpyDeviceToHost);
                if (err != cudaSuccess) {
                    fprintf(stderr, "APIC: Warning - failed to capture data for bound region '%s': %d\n", name, err);
                    region.initial_data.clear();
                }
            }
            break;
        }
    }
}

void wp_apic_register_ptr_location(APICState state, uint32_t region_id, uint64_t offset, uint64_t stride)
{
    if (!state)
        return;

    APICPtrLocation loc;
    loc.region_id = region_id;
    loc.offset = offset;
    loc.stride = stride;
    state->ptr_locations.push_back(loc);
}

// =============================================================================
// Internal Recording Functions (called from wp_cuda_launch_kernel, memcpy, etc.)
// =============================================================================

void apic_record_kernel_launch(
    APICState state,
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
)
{
    if (!state || !state->recording)
        return;

    // Build param bindings data (arrays and scalars)
    std::vector<uint8_t> params_data;
    for (int i = 0; i < num_params; i++) {
        const APICLaunchParam& param = params[i];
        APICLaunchParamRecord rec = {};
        rec.is_array = param.is_array;
        rec.ndim = param.ndim;
        rec.param_index = param.param_index;
        rec.region_id = param.region_id;
        rec.byte_offset = param.byte_offset;
        for (int d = 0; d < APIC_MAX_DIMS; d++) {
            rec.shape[d] = param.shape[d];
            rec.strides[d] = param.strides[d];
        }
        rec.element_size = param.element_size;
        size_t off = params_data.size();
        params_data.resize(off + sizeof(rec));
        memcpy(params_data.data() + off, &rec, sizeof(rec));
    }

    size_t key_len = kernel_key ? strlen(kernel_key) : 0;
    size_t hash_len = module_hash ? strlen(module_hash) : 0;
    uint32_t total_size = sizeof(APICLaunchRecord) + key_len + hash_len + params_data.size();

    // Build launch record with embedded launch bounds
    APICLaunchRecord rec = {};
    rec.header.op_type = APIC_OP_KERNEL_LAUNCH;
    rec.header.total_size = total_size;
    // Embed launch bounds (shape/ndim/size)
    rec.ndim = ndim;
    rec.size = dim;
    for (int d = 0; d < ndim && d < APIC_LAUNCH_MAX_DIMS; d++) {
        rec.shape[d] = shape[d];
    }
    rec.dim = dim;
    rec.max_blocks = max_blocks;
    rec.block_dim = block_dim;
    rec.smem_bytes = smem_bytes;
    rec.is_forward = is_forward ? 1 : 0;
    rec.kernel_key_len = static_cast<uint16_t>(key_len);
    rec.module_hash_len = static_cast<uint16_t>(hash_len);
    rec.num_params = static_cast<uint16_t>(num_params);

    // Append to operation stream
    state->append_bytes(&rec, sizeof(rec));
    if (key_len > 0)
        state->append_bytes(kernel_key, key_len);
    if (hash_len > 0)
        state->append_bytes(module_hash, hash_len);
    if (!params_data.empty())
        state->append_bytes(params_data.data(), params_data.size());

    state->operation_count++;
}

void apic_record_memcpy(APICState state, void* dst, void* src, size_t size, APICOpType kind)
{
    if (!state || !state->recording)
        return;

    // Resolve destination pointer to region
    int32_t dst_region_id = -1;
    uint64_t dst_offset = 0;
    if (!state->find_region(reinterpret_cast<uint64_t>(dst), dst_region_id, dst_offset)) {
        fprintf(stderr, "APIC: Warning - memcpy dst pointer not in any registered region\n");
    }

    if (kind == APIC_OP_MEMCPY_D2D || kind == APIC_OP_MEMCPY_H2H) {
        // Both src and dst are in tracked regions (device memory or host memory)
        int32_t src_region_id = -1;
        uint64_t src_offset = 0;
        if (!state->find_region(reinterpret_cast<uint64_t>(src), src_region_id, src_offset)) {
            fprintf(stderr, "APIC: Warning - memcpy src pointer not in any registered region\n");
        }

        APICMemcpyD2DRecord rec = {};
        rec.header.op_type = static_cast<uint8_t>(kind);  // preserve D2D vs H2H distinction
        rec.header.total_size = sizeof(rec);
        rec.dst_region_id = dst_region_id;
        rec.src_region_id = src_region_id;
        rec.dst_offset = dst_offset;
        rec.src_offset = src_offset;
        rec.size = size;

        state->append_bytes(&rec, sizeof(rec));
    } else if (kind == APIC_OP_MEMCPY_H2D) {
        uint32_t total_size = sizeof(APICMemcpyH2DRecord) + size;

        APICMemcpyH2DRecord rec = {};
        rec.header.op_type = APIC_OP_MEMCPY_H2D;
        rec.header.total_size = total_size;
        rec.dst_region_id = dst_region_id;
        rec.dst_offset = dst_offset;
        rec.size = size;

        state->append_bytes(&rec, sizeof(rec));
        if (src && size > 0) {
            state->append_bytes(src, size);
        }
    }

    state->operation_count++;
}

void apic_record_memset(APICState state, void* dst, int value, size_t size)
{
    if (!state || !state->recording)
        return;

    // Resolve destination pointer to region
    int32_t dst_region_id = -1;
    uint64_t dst_offset = 0;
    if (!state->find_region(reinterpret_cast<uint64_t>(dst), dst_region_id, dst_offset)) {
        fprintf(stderr, "APIC: Warning - memset dst pointer not in any registered region\n");
    }

    APICMemsetRecord rec = {};
    rec.header.op_type = APIC_OP_MEMSET;
    rec.header.total_size = sizeof(rec);
    rec.region_id = dst_region_id;
    rec.value = value;
    rec.offset = dst_offset;
    rec.size = size;

    state->append_bytes(&rec, sizeof(rec));
    state->operation_count++;
}

void apic_record_alloc(APICState state, void* ptr, size_t size)
{
    if (!state || !state->recording)
        return;

    // Register as a memory region with auto-generated ID
    uint32_t region_id = state->next_region_id++;

    APICRegion region;
    region.region_id = region_id;
    region.base_ptr = reinterpret_cast<uint64_t>(ptr);
    region.size = size;
    region.element_size = 1;  // Unknown at alloc time

    state->memory_regions[region.base_ptr] = region;

    // Record the allocation operation directly to stream
    APICAllocRecord rec = {};
    rec.header.op_type = APIC_OP_ALLOC;
    rec.header.total_size = sizeof(rec);
    rec.region_id = region_id;
    rec.size = size;

    state->append_bytes(&rec, sizeof(rec));
    state->operation_count++;
}

// =============================================================================
// APIC WGF File Writing - Serialize directly from APICGraphInternal
// =============================================================================

// Helper: write uint32_t to buffer
static void apic_write_u32(std::vector<uint8_t>& buf, uint32_t val)
{
    size_t off = buf.size();
    buf.resize(off + 4);
    memcpy(buf.data() + off, &val, 4);
}

// Helper: write uint64_t to buffer
static void apic_write_u64(std::vector<uint8_t>& buf, uint64_t val)
{
    size_t off = buf.size();
    buf.resize(off + 8);
    memcpy(buf.data() + off, &val, 8);
}

// Helper: write length-prefixed string to buffer
static void apic_write_string(std::vector<uint8_t>& buf, const std::string& s)
{
    apic_write_u32(buf, static_cast<uint32_t>(s.size()));
    if (!s.empty()) {
        size_t off = buf.size();
        buf.resize(off + s.size());
        memcpy(buf.data() + off, s.data(), s.size());
    }
}

// Helper: find region ID for a device pointer, or register and capture device data
static uint32_t
apic_find_or_register_region_with_data(APICGraphInternal* state, uint64_t ptr, uint64_t size, uint32_t elem_size)
{
    // Check if region already exists
    auto it = state->memory_regions.find(ptr);
    if (it != state->memory_regions.end()) {
        return it->second.region_id;
    }

    // Register new region
    uint32_t region_id = wp_apic_register_memory_region(state, ptr, size, elem_size);

    // Capture data - use memcpy for CPU, cudaMemcpy for CUDA
    auto& region = state->memory_regions[ptr];
    region.initial_data.resize(size);
    if (state->is_cpu) {
        memcpy(region.initial_data.data(), (void*)ptr, size);
    } else {
        cudaError_t err = cudaMemcpy(region.initial_data.data(), (void*)ptr, size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "APIC: Warning - failed to capture device data for region %u: %d\n", region_id, err);
            region.initial_data.clear();
        }
    }

    return region_id;
}

// Collect all meshes for serialization
// All meshes in g_mesh_descriptors are serialized - their arrays will be registered as regions
static std::vector<std::pair<uint64_t, wp::Mesh>> apic_collect_meshes_for_serialization()
{
    std::vector<std::pair<uint64_t, wp::Mesh>> meshes;

    for (const auto& kv : wp::g_mesh_descriptors) {
        meshes.push_back({ kv.first, kv.second });
    }

    return meshes;
}

// Write a single mesh as APICMeshRecord to the buffer
// Registers memory regions for mesh arrays and captures their data
static void
apic_write_mesh(std::vector<uint8_t>& data, APICGraphInternal* state, uint64_t mesh_id, const wp::Mesh& mesh)
{
    // Find or register regions for mesh arrays
    uint64_t points_ptr = (uint64_t)mesh.points.data;
    uint64_t indices_ptr = (uint64_t)mesh.indices.data;
    uint64_t velocities_ptr = (uint64_t)mesh.velocities.data;

    uint64_t points_size = mesh.num_points * sizeof(wp::vec3);
    uint64_t indices_size = mesh.num_tris * 3 * sizeof(int);
    uint64_t velocities_size = mesh.velocities.data ? mesh.num_points * sizeof(wp::vec3) : 0;

    uint32_t points_region_id
        = apic_find_or_register_region_with_data(state, points_ptr, points_size, sizeof(wp::vec3));
    uint32_t indices_region_id = apic_find_or_register_region_with_data(state, indices_ptr, indices_size, sizeof(int));
    uint32_t velocities_region_id = UINT32_MAX;
    if (mesh.velocities.data) {
        velocities_region_id
            = apic_find_or_register_region_with_data(state, velocities_ptr, velocities_size, sizeof(wp::vec3));
    }

    APICMeshRecord rec = {};
    rec.num_points = mesh.num_points;
    rec.num_tris = mesh.num_tris;
    rec.support_winding_number = mesh.solid_angle_props ? 1 : 0;
    rec.bvh_constructor = 0;  // Default SAH constructor
    rec.bvh_leaf_size = 1;  // Default leaf size
    rec.points_region_id = points_region_id;
    rec.indices_region_id = indices_region_id;
    rec.velocities_region_id = velocities_region_id;
    rec.original_ptr = mesh_id;

    size_t off = data.size();
    data.resize(off + sizeof(APICMeshRecord));
    memcpy(data.data() + off, &rec, sizeof(APICMeshRecord));
}

// Build binary metadata section from internal state
static std::vector<uint8_t> apic_serialize_metadata(APICGraphInternal* state, uint32_t target_arch)
{
    std::vector<uint8_t> data;

    // Collect all meshes for serialization
    auto meshes = apic_collect_meshes_for_serialization();

    // Header: version, target_arch, num_modules, num_kernels, num_params, num_meshes, num_ptr_locations
    apic_write_u32(data, APIC_FORMAT_VERSION);
    apic_write_u32(data, target_arch);
    apic_write_u32(data, static_cast<uint32_t>(state->modules.size()));
    apic_write_u32(data, static_cast<uint32_t>(state->kernels.size()));
    apic_write_u32(data, static_cast<uint32_t>(state->recording_bindings.size()));
    apic_write_u32(data, static_cast<uint32_t>(meshes.size()));

    // Pointer locations count
    apic_write_u32(data, static_cast<uint32_t>(state->ptr_locations.size()));

    // Modules: hash, name, cubin_filename, target_arch
    for (const auto& kv : state->modules) {
        const APICModule& m = kv.second;
        apic_write_string(data, m.module_hash);
        apic_write_string(data, m.module_name);
        apic_write_string(data, m.cubin_filename);
        apic_write_u32(data, static_cast<uint32_t>(m.target_arch));
    }

    // Kernels: key, module_hash, forward_name, backward_name, smem bytes, block_dim
    for (const auto& kv : state->kernels) {
        const APICKernel& k = kv.second;
        apic_write_string(data, k.kernel_key);
        apic_write_string(data, k.module_hash);
        apic_write_string(data, k.forward_name);
        apic_write_string(data, k.backward_name);
        apic_write_u32(data, static_cast<uint32_t>(k.forward_smem_bytes));
        apic_write_u32(data, static_cast<uint32_t>(k.backward_smem_bytes));
        apic_write_u32(data, static_cast<uint32_t>(k.block_dim));
    }

    // Params: name, region_id
    for (const auto& b : state->recording_bindings) {
        apic_write_string(data, b.first);
        apic_write_u32(data, b.second);
    }

    // Meshes: written as APICMeshRecord structs
    for (const auto& kv : meshes) {
        apic_write_mesh(data, state, kv.first, kv.second);
    }

    // Pointer locations: region_id, offset, stride
    for (const auto& loc : state->ptr_locations) {
        apic_write_u32(data, loc.region_id);
        apic_write_u64(data, loc.offset);
        apic_write_u64(data, loc.stride);
    }

    return data;
}

int wp_apic_state_save(APICState state, const char* path, uint32_t target_arch)
{
    if (!state) {
        fprintf(stderr, "APIC: Null state passed to wp_apic_state_save\n");
        return 0;
    }

    // Build metadata section from internal state
    std::vector<uint8_t> metadata_section = apic_serialize_metadata(state, target_arch);

    // Build memory section from state->memory_regions
    // Write ALL regions (not just ones with data) so we have size info for input/output params
    std::vector<uint8_t> memory_section;
    {
        uint32_t region_count = static_cast<uint32_t>(state->memory_regions.size());

        // Write count
        memory_section.resize(4);
        memcpy(memory_section.data(), &region_count, 4);

        // Write each region
        for (const auto& kv : state->memory_regions) {
            const APICRegion& region = kv.second;

            APICMemoryRegionRecord rec = {};
            rec.region_id = region.region_id;
            rec.element_size = region.element_size;
            rec.size = region.size;
            rec.has_initial_data = region.initial_data.empty() ? 0 : 1;

            size_t offset = memory_section.size();
            size_t data_size = rec.has_initial_data ? region.initial_data.size() : 0;
            memory_section.resize(offset + sizeof(rec) + data_size);
            memcpy(memory_section.data() + offset, &rec, sizeof(rec));
            if (rec.has_initial_data) {
                memcpy(
                    memory_section.data() + offset + sizeof(rec), region.initial_data.data(), region.initial_data.size()
                );
            }
        }
    }

    // Build operations section - prepend count to the operation stream
    std::vector<uint8_t> ops_section;
    {
        // Write operation count
        ops_section.resize(4 + state->operation_stream.size());
        memcpy(ops_section.data(), &state->operation_count, 4);
        // Copy the entire operation stream (already in serialized format)
        if (!state->operation_stream.empty()) {
            memcpy(ops_section.data() + 4, state->operation_stream.data(), state->operation_stream.size());
        }
    }

    // Write WGF file
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "APIC: Failed to open file for writing: %s\n", path);
        return 0;
    }

    const uint32_t HEADER_SIZE = 64;
    const uint32_t SECTION_ENTRY_SIZE = 32;
    uint32_t num_sections = 3;

    uint64_t section_table_offset = HEADER_SIZE;
    uint64_t data_offset = section_table_offset + num_sections * SECTION_ENTRY_SIZE;
    uint64_t metadata_offset = data_offset;
    uint64_t memory_offset = metadata_offset + metadata_section.size();
    uint64_t operations_offset = memory_offset + memory_section.size();

    // Write header
    APICFileHeader header = {};
    header.magic[0] = 'W';
    header.magic[1] = 'G';
    header.magic[2] = 'F';
    header.magic[3] = '1';
    header.version = APIC_FORMAT_VERSION;
    header.num_sections = num_sections;
    header.section_table_offset = section_table_offset;
    header.target_arch = target_arch;

    if (fwrite(&header, sizeof(header), 1, f) != 1) {
        fclose(f);
        return 0;
    }

    // Write section table
    APICSectionEntry entries[3] = {};
    entries[0].type = APIC_SECTION_METADATA;
    entries[0].offset = metadata_offset;
    entries[0].size = entries[0].uncompressed_size = static_cast<int64_t>(metadata_section.size());
    entries[1].type = APIC_SECTION_MEMORY;
    entries[1].offset = memory_offset;
    entries[1].size = entries[1].uncompressed_size = static_cast<int64_t>(memory_section.size());
    entries[2].type = APIC_SECTION_OPERATIONS;
    entries[2].offset = operations_offset;
    entries[2].size = entries[2].uncompressed_size = static_cast<int64_t>(ops_section.size());

    if (fwrite(entries, sizeof(APICSectionEntry), 3, f) != 3) {
        fclose(f);
        return 0;
    }

    // Write section data
    if (!metadata_section.empty()
        && fwrite(metadata_section.data(), 1, metadata_section.size(), f) != metadata_section.size()) {
        fclose(f);
        return 0;
    }
    if (!memory_section.empty()
        && fwrite(memory_section.data(), 1, memory_section.size(), f) != memory_section.size()) {
        fclose(f);
        return 0;
    }
    if (!ops_section.empty() && fwrite(ops_section.data(), 1, ops_section.size(), f) != ops_section.size()) {
        fclose(f);
        return 0;
    }

    fclose(f);
    return 1;
}

// =============================================================================
// APIC Graph Loading Implementation
// =============================================================================
// Uses POD structs from apic_types.h for version 2 format

// WGF file format constants (aliases to apic_types.h values for compatibility)
static const char WGF_MAGIC[4] = { 'W', 'G', 'F', '1' };
static const uint32_t WGF_VERSION = APIC_FORMAT_VERSION;
static const uint32_t WGF_SECTION_METADATA = APIC_SECTION_METADATA;
static const uint32_t WGF_SECTION_MEMORY = APIC_SECTION_MEMORY;
static const uint32_t WGF_SECTION_OPERATIONS = APIC_SECTION_OPERATIONS;

// Helper: Read file contents
static bool apic_read_file(const char* path, std::vector<uint8_t>& data)
{
    FILE* f = fopen(path, "rb");
    if (!f)
        return false;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    data.resize(size);
    size_t read = fread(data.data(), 1, size, f);
    fclose(f);

    return read == (size_t)size;
}

// Helper: Read string from buffer
static std::string apic_read_string(const uint8_t*& ptr, size_t len)
{
    std::string s(reinterpret_cast<const char*>(ptr), len);
    ptr += len;
    return s;
}

// Helper: Read value from buffer
template <typename T> static T apic_read_value(const uint8_t*& ptr)
{
    T value;
    memcpy(&value, ptr, sizeof(T));
    ptr += sizeof(T);
    return value;
}

// Helper: read length-prefixed string from buffer
static std::string apic_read_lp_string(const uint8_t*& ptr)
{
    uint32_t len = apic_read_value<uint32_t>(ptr);
    std::string s(reinterpret_cast<const char*>(ptr), len);
    ptr += len;
    return s;
}

// Parse binary metadata section
// Format: header (7 x u32) + modules + kernels + params + meshes + ptr_locations
static bool apic_parse_metadata(const uint8_t* data, size_t size, APICGraphInternal* graph)
{
    if (!data || size < 28)  // Minimum header size: 7 x uint32
        return false;

    const uint8_t* ptr = data;

    // Read header
    uint32_t version = apic_read_value<uint32_t>(ptr);
    (void)version;  // Could validate version here
    graph->target_arch = apic_read_value<uint32_t>(ptr);
    uint32_t num_modules = apic_read_value<uint32_t>(ptr);
    uint32_t num_kernels = apic_read_value<uint32_t>(ptr);
    uint32_t num_params = apic_read_value<uint32_t>(ptr);
    uint32_t num_meshes = apic_read_value<uint32_t>(ptr);
    uint32_t num_ptr_locations = apic_read_value<uint32_t>(ptr);

    // Read modules
    for (uint32_t i = 0; i < num_modules; i++) {
        APICModule mod;
        mod.module_hash = apic_read_lp_string(ptr);
        mod.module_name = apic_read_lp_string(ptr);
        mod.cubin_filename = apic_read_lp_string(ptr);
        mod.target_arch = apic_read_value<uint32_t>(ptr);
        mod.cuda_module = nullptr;
        graph->modules[mod.module_hash] = mod;
    }

    // Read kernels
    for (uint32_t i = 0; i < num_kernels; i++) {
        APICKernel info;
        info.kernel_key = apic_read_lp_string(ptr);
        info.module_hash = apic_read_lp_string(ptr);
        info.forward_name = apic_read_lp_string(ptr);
        info.backward_name = apic_read_lp_string(ptr);
        info.forward_smem_bytes = apic_read_value<uint32_t>(ptr);
        info.backward_smem_bytes = apic_read_value<uint32_t>(ptr);
        info.block_dim = apic_read_value<uint32_t>(ptr);
        graph->kernels[info.kernel_key] = info;
    }

    // Read params
    for (uint32_t i = 0; i < num_params; i++) {
        std::string name = apic_read_lp_string(ptr);
        uint32_t region_id = apic_read_value<uint32_t>(ptr);
        graph->bindings[name] = region_id;
        graph->binding_names.push_back(name);
    }

    // Read mesh records (deferred creation until memory is allocated)
    for (uint32_t i = 0; i < num_meshes; i++) {
        APICMeshRecord rec;
        memcpy(&rec, ptr, sizeof(APICMeshRecord));
        ptr += sizeof(APICMeshRecord);
        graph->mesh_records.push_back(rec);
    }

    // Read pointer locations
    for (uint32_t i = 0; i < num_ptr_locations; i++) {
        APICPtrLocation loc;
        loc.region_id = apic_read_value<uint32_t>(ptr);
        loc.offset = apic_read_value<uint64_t>(ptr);
        loc.stride = apic_read_value<uint64_t>(ptr);
        graph->ptr_locations.push_back(loc);
    }

    return true;
}

// Parse operations section - just copy the stream directly
// Operations are stored in serialized format and iterated during rebuild
static bool apic_parse_operations(const uint8_t* data, size_t size, APICGraphInternal* graph)
{
    if (!data || size < 4)
        return false;

    const uint8_t* ptr = data;

    // Read operation count
    graph->operation_count = apic_read_value<uint32_t>(ptr);

    // Copy the rest of the stream (operations data)
    size_t stream_size = size - 4;
    if (stream_size > 0) {
        graph->operation_stream.resize(stream_size);
        memcpy(graph->operation_stream.data(), ptr, stream_size);
    }

    return true;
}

// Parse memory section to create region entries (must be called before allocation)
static bool apic_parse_memory_regions(const uint8_t* data, size_t size, APICGraphInternal* graph)
{
    if (!data || size < 4)
        return true;  // Empty is OK

    const uint8_t* ptr = data;
    const uint8_t* end = data + size;

    uint32_t region_count = apic_read_value<uint32_t>(ptr);

    for (uint32_t i = 0; i < region_count; i++) {
        if (ptr + sizeof(APICMemoryRegionRecord) > end)
            return false;

        const APICMemoryRegionRecord* rec = reinterpret_cast<const APICMemoryRegionRecord*>(ptr);
        ptr += sizeof(APICMemoryRegionRecord);

        // Create region entry if not already in graph
        if (graph->regions.find(rec->region_id) == graph->regions.end()) {
            APICRegion region;
            region.region_id = rec->region_id;
            region.size = rec->size;
            region.element_size = rec->element_size;
            region.ptr = nullptr;  // Will be allocated later
            graph->regions[rec->region_id] = region;
        }

        // Skip initial data if present
        if (rec->has_initial_data) {
            ptr += rec->size;
        }
    }

    return true;
}

// Initialize memory regions with saved data (version 2 format using APICMemoryRegionRecord)
// Must be called AFTER allocation
static bool apic_init_memory(const uint8_t* data, size_t size, APICGraphInternal* graph)
{
    if (!data || size < 4)
        return true;  // Empty is OK

    const uint8_t* ptr = data;
    const uint8_t* end = data + size;

    uint32_t region_count = apic_read_value<uint32_t>(ptr);

    for (uint32_t i = 0; i < region_count; i++) {
        if (ptr + sizeof(APICMemoryRegionRecord) > end)
            return false;

        const APICMemoryRegionRecord* rec = reinterpret_cast<const APICMemoryRegionRecord*>(ptr);
        ptr += sizeof(APICMemoryRegionRecord);

        if (rec->has_initial_data) {
            auto it = graph->regions.find(rec->region_id);
            if (it != graph->regions.end() && it->second.ptr) {
                if (graph->is_cpu) {
                    memcpy(it->second.ptr, ptr, rec->size);
                } else {
                    // Copy data to device using runtime API
                    cudaError_t err = cudaMemcpy(it->second.ptr, ptr, rec->size, cudaMemcpyHostToDevice);
                    if (err != cudaSuccess) {
                        wp::set_error_string("Failed to initialize memory region %u", rec->region_id);
                        return false;
                    }
                }
            }
            ptr += rec->size;
        }
    }

    return true;
}

// Create meshes from stored mesh records after memory regions are allocated
// This populates handle_ptr_remap with old_mesh_ptr -> new_mesh_ptr mappings
static bool apic_create_meshes(APICGraphInternal* graph)
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

// Fixup handle pointers in memory regions after loading
// For each registered pointer location, remap old handle values to new ones
// Note: regions are in device memory, so we must use cudaMemcpy
static void apic_fixup_ptr_locations(APICGraphInternal* graph)
{
    if (graph->handle_ptr_remap.empty() || graph->ptr_locations.empty())
        return;

    for (const auto& loc : graph->ptr_locations) {
        auto region_it = graph->regions.find(loc.region_id);
        if (region_it == graph->regions.end() || !region_it->second.ptr)
            continue;

        uint8_t* base = static_cast<uint8_t*>(region_it->second.ptr);
        uint64_t region_size = region_it->second.size;

        if (loc.stride == 0) {
            // Single pointer at offset
            if (loc.offset + sizeof(uint64_t) <= region_size) {
                uint8_t* device_ptr = base + loc.offset;
                if (graph->is_cpu) {
                    // Direct host memory access
                    uint64_t old_val = *(uint64_t*)device_ptr;
                    auto remap_it = graph->handle_ptr_remap.find(old_val);
                    if (remap_it != graph->handle_ptr_remap.end()) {
                        *(uint64_t*)device_ptr = remap_it->second;
                    }
                } else {
                    uint64_t old_val;
                    cudaMemcpy(&old_val, device_ptr, sizeof(uint64_t), cudaMemcpyDeviceToHost);
                    auto remap_it = graph->handle_ptr_remap.find(old_val);
                    if (remap_it != graph->handle_ptr_remap.end()) {
                        uint64_t new_val = remap_it->second;
                        cudaMemcpy(device_ptr, &new_val, sizeof(uint64_t), cudaMemcpyHostToDevice);
                    }
                }
            }
        } else {
            // Array of pointers with stride
            for (uint64_t off = loc.offset; off + sizeof(uint64_t) <= region_size; off += loc.stride) {
                uint8_t* device_ptr = base + off;
                if (graph->is_cpu) {
                    // Direct host memory access
                    uint64_t old_val = *(uint64_t*)device_ptr;
                    auto remap_it = graph->handle_ptr_remap.find(old_val);
                    if (remap_it != graph->handle_ptr_remap.end()) {
                        *(uint64_t*)device_ptr = remap_it->second;
                    }
                } else {
                    uint64_t old_val;
                    cudaMemcpy(&old_val, device_ptr, sizeof(uint64_t), cudaMemcpyDeviceToHost);
                    auto remap_it = graph->handle_ptr_remap.find(old_val);
                    if (remap_it != graph->handle_ptr_remap.end()) {
                        uint64_t new_val = remap_it->second;
                        cudaMemcpy(device_ptr, &new_val, sizeof(uint64_t), cudaMemcpyHostToDevice);
                    }
                }
            }
        }
    }
}

// Helper: get kernel function (looks up or retrieves from cache)
static CUfunction apic_get_kernel_function(
    APICGraphInternal* graph,
    const char* module_hash,
    size_t hash_len,
    const char* kernel_key,
    size_t key_len,
    bool is_forward
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
static bool apic_rebuild_cuda_graph(APICGraphInternal* graph, CUstream stream)
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
                    // Scalar parameter - value bytes are stored in shape[] and strides[]
                    size_t scalar_size = binding->byte_offset;
                    auto scalar = std::make_unique<uint8_t[]>(scalar_size);

                    // Copy from shape[] (first 32 bytes) and strides[] (next 32 bytes)
                    const uint8_t* shape_bytes = reinterpret_cast<const uint8_t*>(binding->shape);
                    const uint8_t* strides_bytes = reinterpret_cast<const uint8_t*>(binding->strides);

                    size_t first_part = std::min(scalar_size, (size_t)(APIC_MAX_DIMS * sizeof(int64_t)));
                    memcpy(scalar.get(), shape_bytes, first_part);
                    if (scalar_size > first_part) {
                        memcpy(scalar.get() + first_part, strides_bytes, scalar_size - first_part);
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
                wp_array_copy_device(graph->cuda_context, &dst_arr, &src_arr,
                                     rec->dst_type, rec->src_type, rec->elem_size);
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

// Public API implementations

APICGraph wp_apic_load_graph(void* context, const char* path)
{
    if (!path) {
        wp::set_error_string("Path is null");
        return nullptr;
    }

    // Determine CPU vs CUDA mode: NULL context means CPU
    bool is_cpu = (context == nullptr);

    if (!is_cpu) {
        ContextGuard guard(context);
    }

    // Determine base path and construct file paths
    std::string path_str(path);
    std::string wgf_path = path_str;
    std::string base_name = path_str;

    // Add .wgf extension if not present
    if (path_str.length() < 4 || path_str.substr(path_str.length() - 4) != ".wgf") {
        wgf_path = path_str + ".wgf";
    } else {
        base_name = path_str.substr(0, path_str.length() - 4);
    }

    // Construct modules directory path
    size_t last_sep = base_name.find_last_of("/\\");
    std::string dir_path = (last_sep != std::string::npos) ? base_name.substr(0, last_sep + 1) : "";
    std::string name_only = (last_sep != std::string::npos) ? base_name.substr(last_sep + 1) : base_name;
    std::string modules_dir = dir_path + name_only + "_modules";

    // Read WGF file
    std::vector<uint8_t> file_data;
    if (!apic_read_file(wgf_path.c_str(), file_data)) {
        wp::set_error_string("Failed to read file: %s", wgf_path.c_str());
        return nullptr;
    }

    if (file_data.size() < sizeof(APICFileHeader)) {
        wp::set_error_string("Invalid WGF file: too small");
        return nullptr;
    }

    // Parse header using direct struct read
    const APICFileHeader* header = reinterpret_cast<const APICFileHeader*>(file_data.data());

    if (memcmp(header->magic, WGF_MAGIC, 4) != 0) {
        wp::set_error_string("Invalid WGF file: bad magic");
        return nullptr;
    }

    if (header->version > WGF_VERSION) {
        wp::set_error_string("Unsupported WGF version: %u", header->version);
        return nullptr;
    }

    // Create graph object
    APICGraphInternal* graph = new APICGraphInternal();
    graph->cuda_context = context;
    graph->target_arch = header->target_arch;
    graph->base_path = base_name;
    graph->is_cpu = is_cpu;

    // Parse section table using direct struct reads
    const APICSectionEntry* sections
        = reinterpret_cast<const APICSectionEntry*>(file_data.data() + header->section_table_offset);

    const uint8_t* metadata_ptr = nullptr;
    size_t metadata_size = 0;
    const uint8_t* memory_ptr = nullptr;
    size_t memory_size = 0;
    const uint8_t* operations_ptr = nullptr;
    size_t operations_size = 0;

    for (uint32_t i = 0; i < header->num_sections; i++) {
        const APICSectionEntry& section = sections[i];

        if (section.type == WGF_SECTION_METADATA) {
            metadata_ptr = file_data.data() + section.offset;
            metadata_size = section.size;
        } else if (section.type == WGF_SECTION_MEMORY) {
            memory_ptr = file_data.data() + section.offset;
            memory_size = section.size;
        } else if (section.type == WGF_SECTION_OPERATIONS) {
            operations_ptr = file_data.data() + section.offset;
            operations_size = section.size;
        }
    }

    // Parse metadata (binary format)
    if (metadata_ptr && metadata_size > 0) {
        if (!apic_parse_metadata(metadata_ptr, metadata_size, graph)) {
            wp::set_error_string("Failed to parse metadata");
            delete graph;
            return nullptr;
        }
    }

    // Parse memory section to create region entries (needed for allocation)
    if (memory_ptr && !apic_parse_memory_regions(memory_ptr, memory_size, graph)) {
        wp::set_error_string("Failed to parse memory regions");
        delete graph;
        return nullptr;
    }

    if (is_cpu) {
        // CPU path: skip module loading (Python handles it via wp_load_obj + wp_lookup,
        // then calls wp_apic_graph_register_host_function() to register function pointers)

        // Allocate memory regions with malloc
        for (auto& pair : graph->regions) {
            void* host_ptr = malloc(pair.second.size);
            if (!host_ptr) {
                wp::set_error_string("Failed to allocate %llu bytes for CPU region",
                    (unsigned long long)pair.second.size);
                delete graph;
                return nullptr;
            }
            // Zero-initialize (matches cudaMalloc behavior for consistency)
            memset(host_ptr, 0, pair.second.size);
            pair.second.ptr = host_ptr;
        }

        // Populate region_id_to_ptr for resolve_region_ptr()
        for (auto& pair : graph->regions) {
            graph->region_id_to_ptr[pair.first] = (uint64_t)pair.second.ptr;
        }

        // Initialize memory with saved data
        if (memory_ptr && !apic_init_memory(memory_ptr, memory_size, graph)) {
            delete graph;
            return nullptr;
        }

        // Skip mesh creation and handle fixup for CPU (not supported yet)

    } else {
        // CUDA path: load cubin modules
        for (auto& pair : graph->modules) {
            std::string cubin_path = modules_dir + "/" + pair.second.cubin_filename;

// Try with forward slash on Unix, backslash on Windows
#ifdef _WIN32
            std::replace(cubin_path.begin(), cubin_path.end(), '/', '\\');
#endif

            // Use wp_cuda_load_module which handles file loading
            CUmodule cuda_module = (CUmodule)wp_cuda_load_module(context, cubin_path.c_str());
            if (!cuda_module) {
                wp::set_error_string("Failed to load module %s", cubin_path.c_str());
                delete graph;
                return nullptr;
            }
            pair.second.cuda_module = cuda_module;
        }

        // Use ContextGuard to ensure all operations use the correct CUDA context
        {
            ContextGuard guard(context);

            // Allocate memory regions using runtime API
            for (auto& pair : graph->regions) {
                void* device_ptr = nullptr;
                cudaError_t err = cudaMalloc(&device_ptr, pair.second.size);
                if (err != cudaSuccess) {
                    wp::set_error_string("Failed to allocate %llu bytes: %d", (unsigned long long)pair.second.size, err);
                    delete graph;
                    return nullptr;
                }
                pair.second.ptr = device_ptr;
            }

            // Populate region_id_to_ptr for resolve_region_ptr()
            for (auto& pair : graph->regions) {
                graph->region_id_to_ptr[pair.first] = (uint64_t)pair.second.ptr;
            }

            // Initialize memory with saved data
            if (memory_ptr && !apic_init_memory(memory_ptr, memory_size, graph)) {
                delete graph;
                return nullptr;
            }

            // Synchronize to ensure memory is ready before mesh creation
            cudaDeviceSynchronize();
        }

        // Create meshes from serialized data (populates handle_ptr_remap)
        if (!apic_create_meshes(graph)) {
            delete graph;
            return nullptr;
        }

        // Fixup handle pointers in memory regions (e.g., Mesh, Volume, BVH)
        apic_fixup_ptr_locations(graph);
    }

    // Parse operations
    if (operations_ptr && !apic_parse_operations(operations_ptr, operations_size, graph)) {
        wp::set_error_string("Failed to parse operations");
        delete graph;
        return nullptr;
    }

    return graph;
}

void wp_apic_destroy_graph(APICGraph graph)
{
    if (graph) {
        delete graph;
    }
}

int wp_apic_set_param(APICGraph graph, const char* name, const void* data, size_t size)
{
    if (!graph || !name || !data)
        return 0;

    if (!graph->is_cpu) {
        ContextGuard guard(graph->cuda_context);
    }

    // Look up in params
    auto param_it = graph->bindings.find(name);
    if (param_it == graph->bindings.end()) {
        wp::set_error_string("Unknown parameter: %s", name);
        return 0;
    }
    uint32_t region_id = param_it->second;

    auto region_it = graph->regions.find(region_id);
    if (region_it == graph->regions.end() || !region_it->second.ptr) {
        wp::set_error_string("Parameter region not found: %s", name);
        return 0;
    }

    if (size != region_it->second.size) {
        wp::set_error_string(
            "Size mismatch for parameter %s: expected %llu, got %llu", name, (unsigned long long)region_it->second.size,
            (unsigned long long)size
        );
        return 0;
    }

    if (graph->is_cpu) {
        memcpy(region_it->second.ptr, data, size);
    } else {
        // Copy data to the pre-allocated device memory (device-to-device async copy since input is a device pointer)
        cudaError_t err = cudaMemcpyAsync(region_it->second.ptr, data, size, cudaMemcpyDeviceToDevice, 0);
        if (err != cudaSuccess) {
            wp::set_error_string("Failed to copy parameter data: %d", err);
            return 0;
        }
    }

    return 1;
}

void* wp_apic_get_param_ptr(APICGraph graph, const char* name)
{
    if (!graph || !name)
        return nullptr;

    // Look up in params
    auto param_it = graph->bindings.find(name);
    if (param_it == graph->bindings.end())
        return nullptr;

    auto region_it = graph->regions.find(param_it->second);
    if (region_it == graph->regions.end())
        return nullptr;

    return region_it->second.ptr;
}

int wp_apic_get_param(APICGraph graph, const char* name, void* data, size_t size)
{
    if (!graph || !name || !data)
        return 0;

    if (!graph->is_cpu) {
        ContextGuard guard(graph->cuda_context);
    }

    // Look up in params
    auto param_it = graph->bindings.find(name);
    if (param_it == graph->bindings.end()) {
        wp::set_error_string("Unknown parameter: %s", name);
        return 0;
    }
    uint32_t region_id = param_it->second;

    auto region_it = graph->regions.find(region_id);
    if (region_it == graph->regions.end() || !region_it->second.ptr) {
        wp::set_error_string("Parameter region not found: %s", name);
        return 0;
    }

    if (size != region_it->second.size) {
        wp::set_error_string(
            "Size mismatch for parameter %s: expected %llu, got %llu", name, (unsigned long long)region_it->second.size,
            (unsigned long long)size
        );
        return 0;
    }

    if (graph->is_cpu) {
        memcpy(data, region_it->second.ptr, size);
    } else {
        // Copy data from the pre-allocated device memory to the destination (device-to-device async)
        cudaError_t err = cudaMemcpyAsync(data, region_it->second.ptr, size, cudaMemcpyDeviceToDevice, 0);
        if (err != cudaSuccess) {
            wp::set_error_string("Failed to copy parameter data: %d", err);
            return 0;
        }
    }

    return 1;
}

void* wp_apic_get_cuda_graph(APICGraph graph)
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

void* wp_apic_get_cuda_graph_exec(APICGraph graph)
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

int wp_apic_get_num_params(APICGraph graph) { return graph ? (int)graph->binding_names.size() : 0; }

const char* wp_apic_get_param_name(APICGraph graph, int index)
{
    if (!graph || index < 0 || index >= (int)graph->binding_names.size())
        return nullptr;
    return graph->binding_names[index].c_str();
}

size_t wp_apic_get_param_size(APICGraph graph, const char* name)
{
    if (!graph || !name)
        return 0;
    auto it = graph->bindings.find(name);
    if (it == graph->bindings.end())
        return 0;
    auto region_it = graph->regions.find(it->second);
    if (region_it == graph->regions.end())
        return 0;
    return region_it->second.size;
}

// =============================================================================
// APIC CPU Graph Support
// =============================================================================

// The runtime array_t<T> layout matches apic_array_t fields but may have trailing
// padding due to default struct alignment (apic_array_t is #pragma pack(push,1)).
// We use sizeof(wp::array_t<int>) for args reconstruction since that is the actual
// layout the CPU kernels expect.
static constexpr size_t WP_ARRAY_T_SIZE = sizeof(wp::array_t<int>);

int wp_apic_is_recording_active() { return apic_is_recording(g_apic_state) ? 1 : 0; }

void wp_apic_record_host_memcpy(void* dst, void* src, size_t size)
{
    if (apic_is_recording(g_apic_state)) {
        apic_record_memcpy(g_apic_state, dst, src, size, APIC_OP_MEMCPY_H2H);
    }
}

void wp_apic_record_host_memset(void* dst, int value, size_t size)
{
    if (apic_is_recording(g_apic_state)) {
        apic_record_memset(g_apic_state, dst, value, size);
    }
}

void wp_apic_record_host_memtile(void* dst, size_t total_size)
{
    if (apic_is_recording(g_apic_state)) {
        // Record as H2D with inline data snapshot (the tiling has already executed,
        // so dst contains the final tiled pattern that we capture for replay).
        apic_record_memcpy(g_apic_state, dst, dst, total_size, APIC_OP_MEMCPY_H2D);
    }
}

void wp_apic_record_array_copy(void* dst, void* src, int dst_type, int src_type, int elem_size)
{
    if (!apic_is_recording(g_apic_state))
        return;

    APICGraphInternal* state = g_apic_state;

    // Only support regular arrays for now
    if (dst_type != wp::ARRAY_TYPE_REGULAR || src_type != wp::ARRAY_TYPE_REGULAR) {
        fprintf(stderr, "APIC: Warning - array_copy with non-regular array types (%d, %d) not recorded\n",
                dst_type, src_type);
        return;
    }

    const wp::array_t<void>& dst_arr = *static_cast<const wp::array_t<void>*>(dst);
    const wp::array_t<void>& src_arr = *static_cast<const wp::array_t<void>*>(src);

    // Resolve data pointers to region IDs
    int32_t dst_region_id = -1, src_region_id = -1;
    uint64_t dst_offset = 0, src_offset = 0;
    state->find_region(reinterpret_cast<uint64_t>(dst_arr.data), dst_region_id, dst_offset);
    state->find_region(reinterpret_cast<uint64_t>(src_arr.data), src_region_id, src_offset);

    APICArrayCopyRecord rec = {};
    rec.header.op_type = APIC_OP_ARRAY_COPY;
    rec.header.total_size = sizeof(rec);
    rec.dst_region_id = dst_region_id;
    rec.dst_type = dst_type;
    rec.dst_offset = dst_offset;
    rec.dst_ndim = dst_arr.ndim;
    rec.src_region_id = src_region_id;
    rec.src_type = src_type;
    rec.src_offset = src_offset;
    rec.src_ndim = src_arr.ndim;
    rec.elem_size = elem_size;

    for (int d = 0; d < APIC_MAX_DIMS; d++) {
        rec.dst_shape[d] = dst_arr.shape.dims[d];
        rec.dst_strides[d] = dst_arr.strides[d];
        rec.src_shape[d] = src_arr.shape.dims[d];
        rec.src_strides[d] = src_arr.strides[d];
    }

    state->append_bytes(&rec, sizeof(rec));
    state->operation_count++;
}

void wp_apic_set_cpu_mode(APICState state)
{
    if (state)
        state->is_cpu = true;
}

// CPU kernel function pointer types, one per launch dimension.
typedef void (*cpu_fwd_1d_fn_t)(wp::launch_bounds_t<1>, void*);
typedef void (*cpu_fwd_2d_fn_t)(wp::launch_bounds_t<2>, void*);
typedef void (*cpu_fwd_3d_fn_t)(wp::launch_bounds_t<3>, void*);
typedef void (*cpu_fwd_4d_fn_t)(wp::launch_bounds_t<4>, void*);
typedef void (*cpu_bwd_1d_fn_t)(wp::launch_bounds_t<1>, void*, void*);
typedef void (*cpu_bwd_2d_fn_t)(wp::launch_bounds_t<2>, void*, void*);
typedef void (*cpu_bwd_3d_fn_t)(wp::launch_bounds_t<3>, void*, void*);
typedef void (*cpu_bwd_4d_fn_t)(wp::launch_bounds_t<4>, void*, void*);

// Helper: construct launch_bounds_t<N> from recorded shape/size and call function
template <int N>
static void apic_call_cpu_fwd(void* fn, const int* shape, size_t size, void* args)
{
    wp::launch_bounds_t<N> bounds;
    for (int d = 0; d < N; d++) bounds.shape[d] = shape[d];
    bounds.size = size;
    bounds.tiled = false;
    reinterpret_cast<void(*)(wp::launch_bounds_t<N>, void*)>(fn)(bounds, args);
}

template <int N>
static void apic_call_cpu_bwd(void* fn, const int* shape, size_t size, void* args, void* adj_args)
{
    wp::launch_bounds_t<N> bounds;
    for (int d = 0; d < N; d++) bounds.shape[d] = shape[d];
    bounds.size = size;
    bounds.tiled = false;
    reinterpret_cast<void(*)(wp::launch_bounds_t<N>, void*, void*)>(fn)(bounds, args, adj_args);
}

// Helper: extract shape and size from a launch_bounds_t<N> given N
void apic_parse_launch_bounds(const void* bounds, int ndim, int* out_shape, size_t* out_size)
{
    // launch_bounds_t<N> layout: int shape[N], size_t size, bool tiled
    const int* shape_ptr = reinterpret_cast<const int*>(bounds);
    for (int d = 0; d < ndim && d < APIC_LAUNCH_MAX_DIMS; d++)
        out_shape[d] = shape_ptr[d];

    // size follows shape[N] — need to account for alignment padding before size_t
    // shape is N ints; size_t is 8-byte aligned
    size_t shape_bytes = ndim * sizeof(int);
    size_t size_offset = (shape_bytes + sizeof(size_t) - 1) & ~(sizeof(size_t) - 1);
    *out_size = *reinterpret_cast<const size_t*>(reinterpret_cast<const uint8_t*>(bounds) + size_offset);
}

void wp_apic_record_cpu_launch(
    void* kernel_fn,
    void* bounds,
    int ndim,
    const APICLaunchInfo* apic_info)
{
    if (!apic_is_recording(g_apic_state) || !apic_info)
        return;

    int shape[APIC_LAUNCH_MAX_DIMS] = {};
    size_t dim = 0;
    apic_parse_launch_bounds(bounds, ndim, shape, &dim);

    apic_record_kernel_launch(
        g_apic_state,
        kernel_fn,
        dim,
        shape,
        ndim,
        0,     // max_blocks (not used for CPU)
        1,     // block_dim (single thread for CPU)
        0,     // smem_bytes (not used for CPU)
        apic_info->is_forward != 0,
        apic_info->kernel_key,
        apic_info->module_hash,
        apic_info->params,
        apic_info->num_params);
}

void wp_apic_register_host_function(APICState state, const char* kernel_key, void* forward_fn, void* backward_fn)
{
    if (!state || !kernel_key)
        return;
    state->host_functions[std::string(kernel_key)] = { forward_fn, backward_fn };
}

// Build a CPU kernel args struct from APIC parameter records.
// Returns the total size written to args_buf.
// For arrays, builds array_t with resolved data pointer via g->resolve_region_ptr().
// For scalars, copies inline value bytes.
// Works for both recording state and loaded graph (unified struct).
static size_t apic_build_host_args(
    const APICLaunchParamRecord* params,
    int num_params,
    const APICGraphInternal* g,
    uint8_t* args_buf,
    size_t buf_capacity)
{
    size_t offset = 0;

    for (int j = 0; j < num_params; j++) {
        const APICLaunchParamRecord& p = params[j];

        if (p.is_array) {
            // Align to 8 bytes (pointer alignment, matches array_t<T> alignment)
            offset = (offset + 7) & ~(size_t)7;
            if (offset + WP_ARRAY_T_SIZE > buf_capacity)
                return 0;

            // Write array_t<T> fields matching the layout in array.h
            // (NOT the packed apic_array_t -- the runtime struct may have trailing padding)
            wp::array_t<int> arr = {};
            void* resolved = g->resolve_region_ptr(p.region_id, p.byte_offset);
            arr.data = (int*)resolved;
            arr.grad = nullptr;
            for (int d = 0; d < APIC_MAX_DIMS; d++) {
                arr.shape.dims[d] = (int)p.shape[d];
                arr.strides[d] = (int)p.strides[d];
            }
            arr.ndim = p.ndim;
            memcpy(args_buf + offset, &arr, WP_ARRAY_T_SIZE);
            offset += WP_ARRAY_T_SIZE;
        } else {
            // Scalar parameter
            size_t scalar_size = p.byte_offset;
            if (scalar_size == 0)
                continue;

            // Align to natural alignment (at most 8)
            size_t align = 1;
            if (scalar_size >= 8)
                align = 8;
            else if (scalar_size >= 4)
                align = 4;
            else if (scalar_size >= 2)
                align = 2;
            offset = (offset + align - 1) & ~(align - 1);

            if (offset + scalar_size > buf_capacity)
                return 0;

            // Scalar bytes stored in shape[] (first 32B) and strides[] (next 32B)
            const uint8_t* shape_bytes = reinterpret_cast<const uint8_t*>(p.shape);
            const uint8_t* strides_bytes = reinterpret_cast<const uint8_t*>(p.strides);
            size_t first_part = std::min(scalar_size, (size_t)(APIC_MAX_DIMS * sizeof(int64_t)));
            memcpy(args_buf + offset, shape_bytes, first_part);
            if (scalar_size > first_part) {
                memcpy(args_buf + offset + first_part, strides_bytes, scalar_size - first_part);
            }
            offset += scalar_size;
        }
    }

    return offset;
}

// Single internal CPU replay implementation for the unified struct.
// Works for both in-process recording replay and loaded graph replay.
static int apic_replay_cpu_ops(APICGraphInternal* g)
{
    const uint8_t* ptr = g->operation_stream.data();
    const uint8_t* end = ptr + g->operation_stream.size();

    // Temporary buffer for reconstructed kernel args (reused across launches)
    std::vector<uint8_t> args_buf(4096);

    for (uint32_t i = 0; i < g->operation_count && ptr < end; i++) {
        const APICOpHeader* hdr = reinterpret_cast<const APICOpHeader*>(ptr);
        const uint8_t* op_start = ptr;

        switch (hdr->op_type) {
        case APIC_OP_KERNEL_LAUNCH: {
            const APICLaunchRecord* rec = reinterpret_cast<const APICLaunchRecord*>(ptr);
            const uint8_t* var_data = ptr + sizeof(APICLaunchRecord);

            // Parse kernel key
            std::string key_str(reinterpret_cast<const char*>(var_data), rec->kernel_key_len);

            // Look up host function
            auto fn_it = g->host_functions.find(key_str);
            if (fn_it == g->host_functions.end()) {
                wp::set_error_string("Host function not found for CPU replay: %s", key_str.c_str());
                return 0;
            }

            int ndim = rec->ndim;
            int shape[APIC_LAUNCH_MAX_DIMS] = {};
            for (int d = 0; d < ndim && d < APIC_LAUNCH_MAX_DIMS; d++)
                shape[d] = rec->shape[d];
            size_t launch_size = rec->size;

            // Parse parameter bindings
            const uint8_t* params_ptr = var_data + rec->kernel_key_len + rec->module_hash_len;
            const APICLaunchParamRecord* params
                = reinterpret_cast<const APICLaunchParamRecord*>(params_ptr);

            // Ensure args buffer is large enough
            size_t estimated_size = rec->num_params * (WP_ARRAY_T_SIZE + 8);
            if (args_buf.size() < estimated_size)
                args_buf.resize(estimated_size);

            if (rec->is_forward) {
                // Forward pass: all params are forward args
                size_t args_size = apic_build_host_args(
                    params, rec->num_params, g, args_buf.data(), args_buf.size());
                if (args_size == 0 && rec->num_params > 0) {
                    wp::set_error_string("Failed to reconstruct kernel args for: %s", key_str.c_str());
                    return 0;
                }
                void* fwd = fn_it->second.first;
                if (!fwd) {
                    wp::set_error_string("No forward function registered for: %s", key_str.c_str());
                    return 0;
                }
                switch (ndim) {
                case 1: apic_call_cpu_fwd<1>(fwd, shape, launch_size, args_buf.data()); break;
                case 2: apic_call_cpu_fwd<2>(fwd, shape, launch_size, args_buf.data()); break;
                case 3: apic_call_cpu_fwd<3>(fwd, shape, launch_size, args_buf.data()); break;
                case 4: apic_call_cpu_fwd<4>(fwd, shape, launch_size, args_buf.data()); break;
                default:
                    wp::set_error_string("Unsupported launch ndim %d for: %s", ndim, key_str.c_str());
                    return 0;
                }
            } else {
                // Backward pass: params contain forward args followed by adjoint args
                void* bwd = fn_it->second.second;
                if (!bwd) {
                    wp::set_error_string("No backward function registered for: %s", key_str.c_str());
                    return 0;
                }
                int half = rec->num_params / 2;
                size_t fwd_size = apic_build_host_args(
                    params, half, g, args_buf.data(), args_buf.size());
                if (fwd_size == 0 && half > 0) {
                    wp::set_error_string("Failed to reconstruct forward args for backward: %s", key_str.c_str());
                    return 0;
                }
                std::vector<uint8_t> adj_buf(args_buf.size());
                size_t adj_size = apic_build_host_args(
                    params + half, rec->num_params - half, g, adj_buf.data(), adj_buf.size());
                (void)adj_size;
                switch (ndim) {
                case 1: apic_call_cpu_bwd<1>(bwd, shape, launch_size, args_buf.data(), adj_buf.data()); break;
                case 2: apic_call_cpu_bwd<2>(bwd, shape, launch_size, args_buf.data(), adj_buf.data()); break;
                case 3: apic_call_cpu_bwd<3>(bwd, shape, launch_size, args_buf.data(), adj_buf.data()); break;
                case 4: apic_call_cpu_bwd<4>(bwd, shape, launch_size, args_buf.data(), adj_buf.data()); break;
                default:
                    wp::set_error_string("Unsupported launch ndim %d for backward: %s", ndim, key_str.c_str());
                    return 0;
                }
            }
            break;
        }

        case APIC_OP_MEMCPY_H2H:
        case APIC_OP_MEMCPY_D2D: {
            const APICMemcpyD2DRecord* rec = reinterpret_cast<const APICMemcpyD2DRecord*>(ptr);
            void* dst = g->resolve_region_ptr(rec->dst_region_id, rec->dst_offset);
            void* src = g->resolve_region_ptr(rec->src_region_id, rec->src_offset);
            if (dst && src)
                memcpy(dst, src, rec->size);
            break;
        }

        case APIC_OP_MEMCPY_H2D: {
            // H2D with inline data -- copy inline bytes to destination region
            const APICMemcpyH2DRecord* rec = reinterpret_cast<const APICMemcpyH2DRecord*>(ptr);
            void* dst = g->resolve_region_ptr(rec->dst_region_id, rec->dst_offset);
            const void* src_data = ptr + sizeof(APICMemcpyH2DRecord);
            if (dst)
                memcpy(dst, src_data, rec->size);
            break;
        }

        case APIC_OP_MEMSET: {
            const APICMemsetRecord* rec = reinterpret_cast<const APICMemsetRecord*>(ptr);
            void* dst = g->resolve_region_ptr(rec->region_id, rec->offset);
            if (dst) {
                // Use same logic as wp_memset_host
                if ((rec->size % 4) > 0) {
                    memset(dst, rec->value, rec->size);
                } else {
                    const size_t num_words = rec->size / 4;
                    for (size_t w = 0; w < num_words; ++w)
                        ((int*)dst)[w] = rec->value;
                }
            }
            break;
        }

        case APIC_OP_ARRAY_COPY: {
            const APICArrayCopyRecord* rec = reinterpret_cast<const APICArrayCopyRecord*>(ptr);
            void* dst_data = g->resolve_region_ptr(rec->dst_region_id, rec->dst_offset);
            void* src_data = g->resolve_region_ptr(rec->src_region_id, rec->src_offset);
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
                wp_array_copy_host(&dst_arr, &src_arr, rec->dst_type, rec->src_type, rec->elem_size);
            }
            break;
        }

        case APIC_OP_ALLOC:
            // Allocations are pre-existing (recording) or handled during loading
            break;

        default:
            wp::set_error_string("Unknown operation type during CPU replay: %d", hdr->op_type);
            return 0;
        }

        // Advance to next operation
        ptr = op_start + hdr->total_size;
    }

    return 1;
}

int wp_apic_replay_host_ops(APICState state)
{
    if (!state) {
        wp::set_error_string("Null state passed to wp_apic_replay_host_ops");
        return 0;
    }
    return apic_replay_cpu_ops(state);
}

// =============================================================================
// APIC Loaded CPU Graph Support
// =============================================================================

void wp_apic_graph_register_host_function(APICGraph graph, const char* kernel_key, void* forward_fn, void* backward_fn)
{
    if (!graph || !kernel_key)
        return;
    graph->host_functions[std::string(kernel_key)] = { forward_fn, backward_fn };
}

int wp_apic_replay_loaded_host_graph(APICGraph graph)
{
    if (!graph) {
        wp::set_error_string("Null graph passed to wp_apic_replay_loaded_host_graph");
        return 0;
    }

    if (!graph->is_cpu) {
        wp::set_error_string("wp_apic_replay_loaded_host_graph called on a CUDA graph");
        return 0;
    }

    return apic_replay_cpu_ops(graph);
}
