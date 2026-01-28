/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

// This file is included at the end of warp.cu
// It contains all APIC (API Capture) implementation code

// ============================================================================
// APIC (API Capture) Implementation
// ============================================================================

APICState wp_apic_create_state() { return new APICStateInternal(); }

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
    APICRecordedRegion region;
    region.region_id = region_id;
    region.base_ptr = base_ptr;
    region.size = size;
    region.element_size = element_size;

    state->memory_regions[base_ptr] = region;
    return region_id;
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
    const APICParamBindingInfo* params,
    int num_params
)
{
    if (!state || !state->recording)
        return;

    // Build param bindings data (arrays and scalars)
    std::vector<uint8_t> params_data;
    for (int i = 0; i < num_params; i++) {
        const APICParamBindingInfo& param = params[i];
        APICParamBindingRecord rec = {};
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

    if (kind == APIC_OP_MEMCPY_D2D) {
        // Resolve source pointer to region
        int32_t src_region_id = -1;
        uint64_t src_offset = 0;
        if (!state->find_region(reinterpret_cast<uint64_t>(src), src_region_id, src_offset)) {
            fprintf(stderr, "APIC: Warning - memcpy D2D src pointer not in any registered region\n");
        }

        APICMemcpyD2DRecord rec = {};
        rec.header.op_type = APIC_OP_MEMCPY_D2D;
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

    APICRecordedRegion region;
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
// APIC WGF File Writing - Serialize directly from APICStateInternal
// =============================================================================

int wp_apic_state_save(
    APICState state, const char* path, uint32_t target_arch, const char* metadata_json, size_t metadata_len
)
{
    if (!state) {
        fprintf(stderr, "APIC: Null state passed to wp_apic_state_save\n");
        return 0;
    }

    // Build memory section from state->memory_regions
    // Write ALL regions (not just ones with data) so we have size info for input/output bindings
    std::vector<uint8_t> memory_section;
    {
        uint32_t region_count = static_cast<uint32_t>(state->memory_regions.size());

        // Write count
        memory_section.resize(4);
        memcpy(memory_section.data(), &region_count, 4);

        // Write each region
        for (const auto& kv : state->memory_regions) {
            const APICRecordedRegion& region = kv.second;

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
    uint64_t memory_offset = metadata_offset + metadata_len;
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
    entries[0].size = entries[0].uncompressed_size = static_cast<int64_t>(metadata_len);
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
    if (metadata_len > 0 && fwrite(metadata_json, 1, metadata_len, f) != metadata_len) {
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

// Loaded memory region
struct APICLoadedRegion {
    uint32_t region_id;
    uint64_t size;
    uint32_t element_size;
    void* ptr;  // Allocated device pointer
};

// Loaded module
struct APICLoadedModule {
    std::string module_hash;
    std::string module_name;
    std::string cubin_filename;
    int target_arch;
    CUmodule cuda_module;
};

// Kernel info from metadata
struct APICKernelInfo {
    std::string kernel_key;
    std::string module_hash;
    std::string forward_name;
    std::string backward_name;
    int forward_smem_bytes;
    int backward_smem_bytes;
    int block_dim;
};

// Internal graph structure
struct APICGraphInternal {
    void* cuda_context;
    int target_arch;

    // Loaded modules
    std::unordered_map<std::string, APICLoadedModule> modules;

    // Kernel info
    std::unordered_map<std::string, APICKernelInfo> kernels;

    // Memory regions
    std::unordered_map<uint32_t, APICLoadedRegion> regions;

    // Parameter bindings (name -> region_id) - unified for inputs and outputs
    std::unordered_map<std::string, uint32_t> params;
    std::vector<std::string> param_names;  // Ordered list for indexing

    // Operation stream - stored directly in serialized format
    // Iterate through using APICOpHeader to dispatch
    std::vector<uint8_t> operation_stream;
    uint32_t operation_count;

    // CUDA graph (built once on first access)
    CUgraph cuda_graph;
    CUgraphExec cuda_graph_exec;

    // Base path for modules directory
    std::string base_path;

    APICGraphInternal()
        : cuda_context(nullptr)
        , target_arch(0)
        , operation_count(0)
        , cuda_graph(nullptr)
        , cuda_graph_exec(nullptr)
    {
    }

    ~APICGraphInternal()
    {
        // Free CUDA graph resources using runtime API
        if (cuda_graph_exec) {
            cudaGraphExecDestroy((cudaGraphExec_t)cuda_graph_exec);
        }
        if (cuda_graph) {
            cudaGraphDestroy((cudaGraph_t)cuda_graph);
        }
        // Free allocated memory regions using runtime API
        for (auto& pair : regions) {
            if (pair.second.ptr) {
                cudaFree(pair.second.ptr);
            }
        }
        // Unload modules using wrapper function
        for (auto& pair : modules) {
            if (pair.second.cuda_module) {
                cuModuleUnload_f(pair.second.cuda_module);
            }
        }
    }
};

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

// Parse JSON metadata (simplified parser for our known format)
// Returns true on success
static bool apic_parse_metadata(const std::string& json, APICGraphInternal* graph)
{
    // Use a simple approach: parse key parts we need
    // In production, would use a proper JSON library

    // For now, we'll parse the JSON manually since we control the format
    // This is a simplified parser that handles our specific JSON structure

    auto find_value = [&json](const std::string& key) -> std::string {
        std::string search = "\"" + key + "\":";
        size_t pos = json.find(search);
        if (pos == std::string::npos)
            return "";
        pos += search.length();
        while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\n'))
            pos++;
        if (pos >= json.length())
            return "";

        if (json[pos] == '"') {
            // String value
            pos++;
            size_t end = json.find('"', pos);
            if (end == std::string::npos)
                return "";
            return json.substr(pos, end - pos);
        } else if (json[pos] == '{' || json[pos] == '[') {
            // Object or array - find matching brace
            char open = json[pos];
            char close = (open == '{') ? '}' : ']';
            int depth = 1;
            size_t start = pos;
            pos++;
            while (pos < json.length() && depth > 0) {
                if (json[pos] == open)
                    depth++;
                else if (json[pos] == close)
                    depth--;
                pos++;
            }
            return json.substr(start, pos - start);
        } else {
            // Number or other value
            size_t end = pos;
            while (end < json.length() && json[end] != ',' && json[end] != '}' && json[end] != '\n')
                end++;
            std::string val = json.substr(pos, end - pos);
            // Trim whitespace
            while (!val.empty() && (val.back() == ' ' || val.back() == '\t'))
                val.pop_back();
            return val;
        }
    };

    auto find_int = [&find_value](const std::string& key) -> int {
        std::string val = find_value(key);
        return val.empty() ? 0 : std::stoi(val);
    };

    // Parse target_arch
    graph->target_arch = find_int("target_arch");

    // Parse modules
    std::string modules_json = find_value("modules");
    if (!modules_json.empty() && modules_json[0] == '{') {
        // Parse each module entry
        size_t pos = 1;
        while (pos < modules_json.length()) {
            // Find module hash (key)
            size_t key_start = modules_json.find('"', pos);
            if (key_start == std::string::npos)
                break;
            key_start++;
            size_t key_end = modules_json.find('"', key_start);
            if (key_end == std::string::npos)
                break;
            std::string module_hash = modules_json.substr(key_start, key_end - key_start);

            // Find the module object
            size_t obj_start = modules_json.find('{', key_end);
            if (obj_start == std::string::npos)
                break;
            int depth = 1;
            size_t obj_end = obj_start + 1;
            while (obj_end < modules_json.length() && depth > 0) {
                if (modules_json[obj_end] == '{')
                    depth++;
                else if (modules_json[obj_end] == '}')
                    depth--;
                obj_end++;
            }
            std::string module_obj = modules_json.substr(obj_start, obj_end - obj_start);

            APICLoadedModule mod;
            mod.module_hash = module_hash;

            // Parse module fields
            auto find_in_obj = [&module_obj](const std::string& key) -> std::string {
                std::string search = "\"" + key + "\":";
                size_t p = module_obj.find(search);
                if (p == std::string::npos)
                    return "";
                p += search.length();
                while (p < module_obj.length() && (module_obj[p] == ' ' || module_obj[p] == '\n'))
                    p++;
                if (module_obj[p] == '"') {
                    p++;
                    size_t e = module_obj.find('"', p);
                    return module_obj.substr(p, e - p);
                } else {
                    size_t e = p;
                    while (e < module_obj.length() && module_obj[e] != ',' && module_obj[e] != '}')
                        e++;
                    std::string v = module_obj.substr(p, e - p);
                    while (!v.empty() && (v.back() == ' ' || v.back() == '\t'))
                        v.pop_back();
                    return v;
                }
            };

            mod.module_name = find_in_obj("name");
            mod.cubin_filename = find_in_obj("cubin_filename");
            std::string arch_str = find_in_obj("target_arch");
            mod.target_arch = arch_str.empty() ? 0 : std::stoi(arch_str);
            mod.cuda_module = nullptr;

            graph->modules[module_hash] = mod;

            pos = obj_end;
        }
    }

    // Parse kernels
    std::string kernels_json = find_value("kernels");
    if (!kernels_json.empty() && kernels_json[0] == '{') {
        size_t pos = 1;
        while (pos < kernels_json.length()) {
            size_t key_start = kernels_json.find('"', pos);
            if (key_start == std::string::npos)
                break;
            key_start++;
            size_t key_end = kernels_json.find('"', key_start);
            if (key_end == std::string::npos)
                break;
            std::string kernel_key = kernels_json.substr(key_start, key_end - key_start);

            size_t obj_start = kernels_json.find('{', key_end);
            if (obj_start == std::string::npos)
                break;
            int depth = 1;
            size_t obj_end = obj_start + 1;
            while (obj_end < kernels_json.length() && depth > 0) {
                if (kernels_json[obj_end] == '{')
                    depth++;
                else if (kernels_json[obj_end] == '}')
                    depth--;
                obj_end++;
            }
            std::string kernel_obj = kernels_json.substr(obj_start, obj_end - obj_start);

            APICKernelInfo info;
            info.kernel_key = kernel_key;

            auto find_in_obj = [&kernel_obj](const std::string& key) -> std::string {
                std::string search = "\"" + key + "\":";
                size_t p = kernel_obj.find(search);
                if (p == std::string::npos)
                    return "";
                p += search.length();
                while (p < kernel_obj.length() && (kernel_obj[p] == ' ' || kernel_obj[p] == '\n'))
                    p++;
                if (kernel_obj[p] == '"') {
                    p++;
                    size_t e = kernel_obj.find('"', p);
                    return kernel_obj.substr(p, e - p);
                } else if (kernel_obj.substr(p, 4) == "null") {
                    return "";
                } else {
                    size_t e = p;
                    while (e < kernel_obj.length() && kernel_obj[e] != ',' && kernel_obj[e] != '}')
                        e++;
                    std::string v = kernel_obj.substr(p, e - p);
                    while (!v.empty() && (v.back() == ' ' || v.back() == '\t'))
                        v.pop_back();
                    return v;
                }
            };

            info.module_hash = find_in_obj("module_hash");
            info.forward_name = find_in_obj("forward_name");
            info.backward_name = find_in_obj("backward_name");
            std::string smem_str = find_in_obj("forward_smem_bytes");
            info.forward_smem_bytes = smem_str.empty() ? 0 : std::stoi(smem_str);
            smem_str = find_in_obj("backward_smem_bytes");
            info.backward_smem_bytes = smem_str.empty() ? 0 : std::stoi(smem_str);
            std::string block_str = find_in_obj("block_dim");
            info.block_dim = block_str.empty() ? 256 : std::stoi(block_str);

            graph->kernels[kernel_key] = info;

            pos = obj_end;
        }
    }

    // Parse memory_regions
    std::string regions_json = find_value("memory_regions");
    if (!regions_json.empty() && regions_json[0] == '{') {
        size_t pos = 1;
        while (pos < regions_json.length()) {
            size_t key_start = regions_json.find('"', pos);
            if (key_start == std::string::npos)
                break;
            key_start++;
            size_t key_end = regions_json.find('"', key_start);
            if (key_end == std::string::npos)
                break;
            std::string region_id_str = regions_json.substr(key_start, key_end - key_start);
            uint32_t region_id = std::stoul(region_id_str);

            size_t obj_start = regions_json.find('{', key_end);
            if (obj_start == std::string::npos)
                break;
            int depth = 1;
            size_t obj_end = obj_start + 1;
            while (obj_end < regions_json.length() && depth > 0) {
                if (regions_json[obj_end] == '{')
                    depth++;
                else if (regions_json[obj_end] == '}')
                    depth--;
                obj_end++;
            }
            std::string region_obj = regions_json.substr(obj_start, obj_end - obj_start);

            APICLoadedRegion region;
            region.region_id = region_id;
            region.ptr = nullptr;

            auto find_in_obj = [&region_obj](const std::string& key) -> std::string {
                std::string search = "\"" + key + "\":";
                size_t p = region_obj.find(search);
                if (p == std::string::npos)
                    return "";
                p += search.length();
                while (p < region_obj.length() && (region_obj[p] == ' ' || region_obj[p] == '\n'))
                    p++;
                if (region_obj[p] == '"') {
                    p++;
                    size_t e = region_obj.find('"', p);
                    return region_obj.substr(p, e - p);
                } else {
                    size_t e = p;
                    while (e < region_obj.length() && region_obj[e] != ',' && region_obj[e] != '}')
                        e++;
                    std::string v = region_obj.substr(p, e - p);
                    while (!v.empty() && (v.back() == ' ' || v.back() == '\t'))
                        v.pop_back();
                    return v;
                }
            };

            std::string size_str = find_in_obj("size");
            region.size = size_str.empty() ? 0 : std::stoull(size_str);
            std::string elem_str = find_in_obj("element_size");
            region.element_size = elem_str.empty() ? 0 : std::stoul(elem_str);

            graph->regions[region_id] = region;

            pos = obj_end;
        }
    }

    // Helper lambda to parse a bindings object and add to params
    auto parse_bindings = [&](const std::string& key) {
        std::string bindings_json = find_value(key);
        if (!bindings_json.empty() && bindings_json[0] == '{') {
            size_t pos = 1;
            while (pos < bindings_json.length()) {
                size_t key_start = bindings_json.find('"', pos);
                if (key_start == std::string::npos)
                    break;
                key_start++;
                size_t key_end = bindings_json.find('"', key_start);
                if (key_end == std::string::npos)
                    break;
                std::string name = bindings_json.substr(key_start, key_end - key_start);

                size_t val_start = bindings_json.find(':', key_end);
                if (val_start == std::string::npos)
                    break;
                val_start++;
                while (val_start < bindings_json.length() && bindings_json[val_start] == ' ')
                    val_start++;
                size_t val_end = val_start;
                while (val_end < bindings_json.length() && bindings_json[val_end] != ','
                       && bindings_json[val_end] != '}')
                    val_end++;
                std::string val_str = bindings_json.substr(val_start, val_end - val_start);
                while (!val_str.empty() && (val_str.back() == ' ' || val_str.back() == '\t'))
                    val_str.pop_back();
                uint32_t region_id = std::stoul(val_str);

                graph->params[name] = region_id;
                graph->param_names.push_back(name);

                pos = val_end;
            }
        }
    };

    // Parse input_bindings and output_bindings into unified params map
    parse_bindings("input_bindings");
    parse_bindings("output_bindings");

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
            APICLoadedRegion region;
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
                // Copy data to device using runtime API
                cudaError_t err = cudaMemcpy(it->second.ptr, ptr, rec->size, cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    wp::set_error_string("Failed to initialize memory region %u", rec->region_id);
                    return false;
                }
            }
            ptr += rec->size;
        }
    }

    return true;
}

// Helper: resolve region_id + offset to a pointer
static void* apic_resolve_region_ptr(APICGraphInternal* graph, int32_t region_id, uint64_t offset)
{
    if (region_id < 0)
        return nullptr;
    auto it = graph->regions.find(region_id);
    if (it != graph->regions.end() && it->second.ptr) {
        return (void*)((uint8_t*)it->second.ptr + offset);
    }
    return nullptr;
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

            // Create launch_bounds_t as param[0] from embedded data in record
            auto bounds = std::make_unique<uint8_t[]>(sizeof(apic_launch_bounds_t));
            apic_launch_bounds_t* bounds_ptr = reinterpret_cast<apic_launch_bounds_t*>(bounds.get());
            memset(bounds_ptr, 0, sizeof(apic_launch_bounds_t));
            bounds_ptr->ndim = rec->ndim;
            bounds_ptr->size = rec->size;
            for (int d = 0; d < rec->ndim && d < APIC_LAUNCH_MAX_DIMS; d++) {
                bounds_ptr->shape[d] = rec->shape[d];
            }
            args.push_back(bounds_ptr);
            arg_storage.push_back(std::move(bounds));

            // Parse param bindings (arrays and scalars, starting from param_index 1)
            for (uint16_t j = 0; j < rec->num_params; j++) {
                const APICParamBindingRecord* binding = reinterpret_cast<const APICParamBindingRecord*>(params_ptr);
                params_ptr += sizeof(APICParamBindingRecord);

                if (binding->is_array) {
                    // Array parameter - create array_t structure
                    auto arr = std::make_unique<uint8_t[]>(sizeof(apic_array_t));
                    apic_array_t* arr_ptr = reinterpret_cast<apic_array_t*>(arr.get());
                    memset(arr_ptr, 0, sizeof(apic_array_t));

                    void* resolved = apic_resolve_region_ptr(graph, binding->region_id, binding->byte_offset);
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
                    // byte_offset contains the scalar size
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
            void* dst_ptr = apic_resolve_region_ptr(graph, rec->dst_region_id, rec->dst_offset);
            cuda_err = cudaMemcpyAsync(dst_ptr, src_data, rec->size, cudaMemcpyHostToDevice, (cudaStream_t)stream);
            if (cuda_err != cudaSuccess) {
                wp::set_error_string("Failed H2D memcpy: %d", cuda_err);
                success = false;
            }
            break;
        }

        case APIC_OP_MEMCPY_D2D: {
            const APICMemcpyD2DRecord* rec = reinterpret_cast<const APICMemcpyD2DRecord*>(ptr);
            void* dst_ptr = apic_resolve_region_ptr(graph, rec->dst_region_id, rec->dst_offset);
            void* src_ptr = apic_resolve_region_ptr(graph, rec->src_region_id, rec->src_offset);
            cuda_err = cudaMemcpyAsync(dst_ptr, src_ptr, rec->size, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
            if (cuda_err != cudaSuccess) {
                wp::set_error_string("Failed D2D memcpy: %d", cuda_err);
                success = false;
            }
            break;
        }

        case APIC_OP_MEMSET: {
            const APICMemsetRecord* rec = reinterpret_cast<const APICMemsetRecord*>(ptr);
            void* dst_ptr = apic_resolve_region_ptr(graph, rec->region_id, rec->offset);
            cuda_err = cudaMemsetAsync(dst_ptr, rec->value, rec->size, (cudaStream_t)stream);
            if (cuda_err != cudaSuccess) {
                wp::set_error_string("Failed memset: %d", cuda_err);
                success = false;
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

    ContextGuard guard(context);

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

    // Parse metadata
    if (metadata_ptr && metadata_size > 0) {
        std::string metadata_json(reinterpret_cast<const char*>(metadata_ptr), metadata_size);
        if (!apic_parse_metadata(metadata_json, graph)) {
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

    // Load cubin modules using the existing warp API
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

    // Initialize memory with saved data
    if (memory_ptr && !apic_init_memory(memory_ptr, memory_size, graph)) {
        delete graph;
        return nullptr;
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

    ContextGuard guard(graph->cuda_context);

    // Look up in params
    auto param_it = graph->params.find(name);
    if (param_it == graph->params.end()) {
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

    // Copy data to the pre-allocated device memory (device-to-device async copy since input is a device pointer)
    cudaError_t err = cudaMemcpyAsync(region_it->second.ptr, data, size, cudaMemcpyDeviceToDevice, 0);
    if (err != cudaSuccess) {
        wp::set_error_string("Failed to copy parameter data: %d", err);
        return 0;
    }

    return 1;
}

void* wp_apic_get_param_ptr(APICGraph graph, const char* name)
{
    if (!graph || !name)
        return nullptr;

    // Look up in params
    auto param_it = graph->params.find(name);
    if (param_it == graph->params.end())
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

    ContextGuard guard(graph->cuda_context);

    // Look up in params
    auto param_it = graph->params.find(name);
    if (param_it == graph->params.end()) {
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

    // Copy data from the pre-allocated device memory to the destination (device-to-device async)
    cudaError_t err = cudaMemcpyAsync(data, region_it->second.ptr, size, cudaMemcpyDeviceToDevice, 0);
    if (err != cudaSuccess) {
        wp::set_error_string("Failed to copy parameter data: %d", err);
        return 0;
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

int wp_apic_get_num_params(APICGraph graph) { return graph ? (int)graph->param_names.size() : 0; }

const char* wp_apic_get_param_name(APICGraph graph, int index)
{
    if (!graph || index < 0 || index >= (int)graph->param_names.size())
        return nullptr;
    return graph->param_names[index].c_str();
}

size_t wp_apic_get_param_size(APICGraph graph, const char* name)
{
    if (!graph || !name)
        return 0;
    auto it = graph->params.find(name);
    if (it == graph->params.end())
        return 0;
    auto region_it = graph->regions.find(it->second);
    if (region_it == graph->regions.end())
        return 0;
    return region_it->second.size;
}
