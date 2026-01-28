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

// Internal recording structs (use raw pointers, converted to region_ids during serialization)
struct APICRecordMemcpy {
    uint64_t dst_ptr;
    uint64_t src_ptr;
    uint64_t size;
};

struct APICRecordMemset {
    uint64_t dst_ptr;
    int32_t value;
    uint32_t _pad;
    uint64_t size;
};

struct APICRecordAlloc {
    uint64_t ptr;
    uint64_t size;
};

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
        state->operations.clear();
        state->memory_regions.clear();
        state->kernel_names.clear();
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

uint32_t wp_apic_register_memory_region(
    APICState state, uint64_t base_ptr, uint64_t size, uint32_t element_size, APICMemoryRole role
)
{
    if (!state)
        return UINT32_MAX;

    // Check if region already exists
    auto it = state->memory_regions.find(base_ptr);
    if (it != state->memory_regions.end()) {
        // Update role if needed (e.g., internal -> input)
        if (role > it->second.role) {
            it->second.role = role;
        }
        return it->second.region_id;
    }

    // Create new region
    uint32_t region_id = state->next_region_id++;
    APICMemoryRegion region;
    region.region_id = region_id;
    region.base_ptr = base_ptr;
    region.size = size;
    region.element_size = element_size;
    region.role = role;

    state->memory_regions[base_ptr] = region;
    return region_id;
}

size_t wp_apic_get_num_operations(APICState state)
{
    if (!state)
        return 0;
    return state->operations.size();
}

size_t wp_apic_get_num_memory_regions(APICState state)
{
    if (!state)
        return 0;
    return state->memory_regions.size();
}

size_t wp_apic_get_num_kernels(APICState state)
{
    if (!state)
        return 0;
    return state->kernel_names.size();
}

size_t wp_apic_get_operations_data_size(APICState state)
{
    if (!state)
        return 0;

    size_t total = 0;
    for (const auto& op : state->operations) {
        // Header: type (1 byte) + data size (4 bytes) + data
        total += 1 + 4 + op.data.size();
    }
    return total;
}

void wp_apic_get_operations_data(APICState state, void* buffer, size_t buffer_size)
{
    if (!state || !buffer)
        return;

    uint8_t* ptr = static_cast<uint8_t*>(buffer);
    uint8_t* end = ptr + buffer_size;

    for (const auto& op : state->operations) {
        size_t op_size = 1 + 4 + op.data.size();
        if (ptr + op_size > end)
            break;

        // Write type
        *ptr++ = static_cast<uint8_t>(op.type);

        // Write data size (little-endian)
        uint32_t data_size = static_cast<uint32_t>(op.data.size());
        memcpy(ptr, &data_size, 4);
        ptr += 4;

        // Write data
        if (!op.data.empty()) {
            memcpy(ptr, op.data.data(), op.data.size());
            ptr += op.data.size();
        }
    }
}

size_t wp_apic_get_memory_regions_data_size(APICState state)
{
    if (!state)
        return 0;
    // Each region: region_id (4) + base_ptr (8) + size (8) + element_size (4) + role (1) = 25 bytes
    return state->memory_regions.size() * 25;
}

void wp_apic_get_memory_regions_data(APICState state, void* buffer, size_t buffer_size)
{
    if (!state || !buffer)
        return;

    uint8_t* ptr = static_cast<uint8_t*>(buffer);
    uint8_t* end = ptr + buffer_size;

    for (const auto& pair : state->memory_regions) {
        if (ptr + 25 > end)
            break;

        const APICMemoryRegion& region = pair.second;

        memcpy(ptr, &region.region_id, 4);
        ptr += 4;
        memcpy(ptr, &region.base_ptr, 8);
        ptr += 8;
        memcpy(ptr, &region.size, 8);
        ptr += 8;
        memcpy(ptr, &region.element_size, 4);
        ptr += 4;
        *ptr++ = static_cast<uint8_t>(region.role);
    }
}

size_t wp_apic_get_kernel_names_size(APICState state)
{
    if (!state)
        return 0;

    size_t total = 0;
    for (const auto& name : state->kernel_names) {
        total += name.size() + 1;  // Include null terminator
    }
    return total;
}

void wp_apic_get_kernel_names(APICState state, char* buffer, size_t buffer_size)
{
    if (!state || !buffer)
        return;

    char* ptr = buffer;
    char* end = buffer + buffer_size;

    for (const auto& name : state->kernel_names) {
        size_t len = name.size() + 1;
        if (ptr + len > end)
            break;
        memcpy(ptr, name.c_str(), len);
        ptr += len;
    }
}

// Internal recording functions

void apic_record_kernel_launch(
    APICState state,
    void* kernel,
    size_t dim,
    int max_blocks,
    int block_dim,
    int smem_bytes,
    void** args,
    size_t num_args,
    size_t* arg_sizes
)
{
    if (!state || !state->recording)
        return;

    // Get kernel name
    std::string kernel_name;
    auto it = g_kernel_names.find((CUfunction)kernel);
    if (it != g_kernel_names.end()) {
        kernel_name = it->second;
        state->kernel_names.insert(kernel_name);
    }

    // Create operation
    APICOperation op;
    op.type = APIC_OP_KERNEL_LAUNCH;

    // Serialize launch data
    // Format: kernel_name_len (4) + kernel_name + dim (8) + max_blocks (4) + block_dim (4) + smem_bytes (4)
    //         + num_params (4) + [param_size (4) + param_data]...
    size_t name_len = kernel_name.size();
    size_t total_param_size = 0;
    for (size_t i = 0; i < num_args; i++) {
        total_param_size += 4 + arg_sizes[i];  // size prefix + data
    }

    size_t data_size = 4 + name_len + 8 + 4 + 4 + 4 + 4 + total_param_size;
    op.data.resize(data_size);
    uint8_t* ptr = op.data.data();

    // Write kernel name
    uint32_t name_len32 = static_cast<uint32_t>(name_len);
    memcpy(ptr, &name_len32, 4);
    ptr += 4;
    if (name_len > 0) {
        memcpy(ptr, kernel_name.c_str(), name_len);
        ptr += name_len;
    }

    // Write launch params
    memcpy(ptr, &dim, 8);
    ptr += 8;
    memcpy(ptr, &max_blocks, 4);
    ptr += 4;
    memcpy(ptr, &block_dim, 4);
    ptr += 4;
    memcpy(ptr, &smem_bytes, 4);
    ptr += 4;

    // Write num_params
    uint32_t num_params32 = static_cast<uint32_t>(num_args);
    memcpy(ptr, &num_params32, 4);
    ptr += 4;

    // Write each parameter
    for (size_t i = 0; i < num_args; i++) {
        uint32_t arg_size32 = static_cast<uint32_t>(arg_sizes[i]);
        memcpy(ptr, &arg_size32, 4);
        ptr += 4;
        if (args[i] && arg_sizes[i] > 0) {
            memcpy(ptr, args[i], arg_sizes[i]);
            ptr += arg_sizes[i];
        }
    }

    state->operations.push_back(std::move(op));
}

void apic_record_memcpy(APICState state, void* dst, void* src, size_t size, APICOpType kind)
{
    if (!state || !state->recording)
        return;

    APICRecordMemcpy rec;
    rec.dst_ptr = reinterpret_cast<uint64_t>(dst);
    rec.src_ptr = reinterpret_cast<uint64_t>(src);
    rec.size = size;

    APICOperation op;
    op.type = kind;
    op.data.resize(sizeof(rec));
    memcpy(op.data.data(), &rec, sizeof(rec));
    state->operations.push_back(std::move(op));
}

void apic_record_memset(APICState state, void* dst, int value, size_t size)
{
    if (!state || !state->recording)
        return;

    APICRecordMemset rec;
    rec.dst_ptr = reinterpret_cast<uint64_t>(dst);
    rec.value = value;
    rec._pad = 0;
    rec.size = size;

    APICOperation op;
    op.type = APIC_OP_MEMSET;
    op.data.resize(sizeof(rec));
    memcpy(op.data.data(), &rec, sizeof(rec));
    state->operations.push_back(std::move(op));
}

void apic_record_alloc(APICState state, void* ptr, size_t size)
{
    if (!state || !state->recording)
        return;

    APICRecordAlloc rec;
    rec.ptr = reinterpret_cast<uint64_t>(ptr);
    rec.size = size;

    APICOperation op;
    op.type = APIC_OP_ALLOC;
    op.data.resize(sizeof(rec));
    memcpy(op.data.data(), &rec, sizeof(rec));
    state->operations.push_back(std::move(op));
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
static const size_t WGF_HEADER_SIZE = 64;

// Parameter binding for loaded graphs
struct APICParamBinding {
    uint8_t type;  // 1=array, 2=scalar
    uint16_t param_index;
    int32_t region_id;  // -1 if null
    uint64_t byte_offset;
    int32_t ndim;
    int64_t shape[APIC_MAX_DIMS];
    int64_t strides[APIC_MAX_DIMS];
    uint32_t element_size;
    // For scalars:
    std::vector<uint8_t> scalar_value;
    // Resolved pointer (updated when bindings change)
    void* resolved_ptr;
};

// Kernel launch operation
struct APICKernelLaunch {
    std::string kernel_key;
    std::string module_hash;
    std::string kernel_name;
    uint64_t dim;
    int32_t max_blocks;
    int32_t block_dim;
    int32_t smem_bytes;
    bool is_forward;
    std::vector<APICParamBinding> param_bindings;
    CUfunction kernel_func;  // Resolved kernel function
};

// Memory operation
struct APICMemoryOp {
    uint8_t type;  // 2=H2D, 4=D2D, 5=memset
    int32_t dst_region_id;
    uint64_t dst_offset;
    int32_t src_region_id;  // For D2D
    uint64_t src_offset;  // For D2D
    uint64_t size;
    int32_t value;  // For memset
    std::vector<uint8_t> src_data;  // For H2D
    // Resolved pointers
    void* dst_ptr;
    void* src_ptr;
};

// Operation entry (preserves order)
struct APICOperationEntry {
    bool is_launch;  // true=launch, false=memop
    size_t index;  // Index into launches or memory_ops
};

// Loaded memory region
struct APICLoadedRegion {
    uint32_t region_id;
    uint64_t size;
    uint32_t element_size;
    uint8_t role;
    void* ptr;  // Allocated device pointer
    bool external;  // True if bound to external array
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

    // Input/output bindings (name -> region_id)
    std::unordered_map<std::string, uint32_t> input_bindings;
    std::unordered_map<std::string, uint32_t> output_bindings;
    std::vector<std::string> input_names;  // Ordered list for indexing
    std::vector<std::string> output_names;

    // Operations
    std::vector<APICKernelLaunch> launches;
    std::vector<APICMemoryOp> memory_ops;
    std::vector<APICOperationEntry> operations;

    // CUDA graph
    CUgraph cuda_graph;
    CUgraphExec cuda_graph_exec;
    bool needs_rebuild;

    // Base path for modules directory
    std::string base_path;

    APICGraphInternal()
        : cuda_context(nullptr)
        , target_arch(0)
        , cuda_graph(nullptr)
        , cuda_graph_exec(nullptr)
        , needs_rebuild(true)
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
            if (pair.second.ptr && !pair.second.external) {
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
            region.external = false;

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
            std::string role_str = find_in_obj("role");
            if (role_str == "input" || role_str == "INPUT")
                region.role = APIC_ROLE_INPUT;
            else if (role_str == "output" || role_str == "OUTPUT")
                region.role = APIC_ROLE_OUTPUT;
            else if (role_str == "input_output" || role_str == "INPUT_OUTPUT")
                region.role = APIC_ROLE_INPUT_OUTPUT;
            else
                region.role = APIC_ROLE_INTERNAL;

            graph->regions[region_id] = region;

            pos = obj_end;
        }
    }

    // Parse input_bindings
    std::string inputs_json = find_value("input_bindings");
    if (!inputs_json.empty() && inputs_json[0] == '{') {
        size_t pos = 1;
        while (pos < inputs_json.length()) {
            size_t key_start = inputs_json.find('"', pos);
            if (key_start == std::string::npos)
                break;
            key_start++;
            size_t key_end = inputs_json.find('"', key_start);
            if (key_end == std::string::npos)
                break;
            std::string name = inputs_json.substr(key_start, key_end - key_start);

            size_t val_start = inputs_json.find(':', key_end);
            if (val_start == std::string::npos)
                break;
            val_start++;
            while (val_start < inputs_json.length() && inputs_json[val_start] == ' ')
                val_start++;
            size_t val_end = val_start;
            while (val_end < inputs_json.length() && inputs_json[val_end] != ',' && inputs_json[val_end] != '}')
                val_end++;
            std::string val_str = inputs_json.substr(val_start, val_end - val_start);
            while (!val_str.empty() && (val_str.back() == ' ' || val_str.back() == '\t'))
                val_str.pop_back();
            uint32_t region_id = std::stoul(val_str);

            graph->input_bindings[name] = region_id;
            graph->input_names.push_back(name);

            pos = val_end;
        }
    }

    // Parse output_bindings
    std::string outputs_json = find_value("output_bindings");
    if (!outputs_json.empty() && outputs_json[0] == '{') {
        size_t pos = 1;
        while (pos < outputs_json.length()) {
            size_t key_start = outputs_json.find('"', pos);
            if (key_start == std::string::npos)
                break;
            key_start++;
            size_t key_end = outputs_json.find('"', key_start);
            if (key_end == std::string::npos)
                break;
            std::string name = outputs_json.substr(key_start, key_end - key_start);

            size_t val_start = outputs_json.find(':', key_end);
            if (val_start == std::string::npos)
                break;
            val_start++;
            while (val_start < outputs_json.length() && inputs_json[val_start] == ' ')
                val_start++;
            size_t val_end = val_start;
            while (val_end < outputs_json.length() && outputs_json[val_end] != ',' && outputs_json[val_end] != '}')
                val_end++;
            std::string val_str = outputs_json.substr(val_start, val_end - val_start);
            while (!val_str.empty() && (val_str.back() == ' ' || val_str.back() == '\t'))
                val_str.pop_back();
            uint32_t region_id = std::stoul(val_str);

            graph->output_bindings[name] = region_id;
            graph->output_names.push_back(name);

            pos = val_end;
        }
    }

    return true;
}

// Parse parameter binding from operations data (version 2 format using fixed-size structs)
static bool apic_parse_param_binding(const uint8_t*& ptr, const uint8_t* end, APICParamBinding& binding)
{
    if (ptr >= end)
        return false;

    // Peek at the type to determine which struct to read
    uint8_t param_type = *ptr;

    if (param_type == APIC_PARAM_ARRAY) {
        // Read fixed-size APICArrayBindingRecord (88 bytes)
        if (ptr + sizeof(APICArrayBindingRecord) > end)
            return false;

        const APICArrayBindingRecord* rec = reinterpret_cast<const APICArrayBindingRecord*>(ptr);
        ptr += sizeof(APICArrayBindingRecord);

        binding.type = rec->type;
        binding.param_index = rec->param_index;
        binding.region_id = rec->region_id;
        binding.byte_offset = rec->byte_offset;
        binding.ndim = rec->ndim;
        binding.element_size = rec->element_size;

        for (int i = 0; i < APIC_MAX_DIMS; i++) {
            binding.shape[i] = rec->shape[i];
            binding.strides[i] = rec->strides[i];
        }
        binding.resolved_ptr = nullptr;

    } else if (param_type == APIC_PARAM_SCALAR) {
        // Read fixed-size APICScalarBindingRecord (136 bytes)
        if (ptr + sizeof(APICScalarBindingRecord) > end)
            return false;

        const APICScalarBindingRecord* rec = reinterpret_cast<const APICScalarBindingRecord*>(ptr);
        ptr += sizeof(APICScalarBindingRecord);

        binding.type = rec->type;
        binding.param_index = rec->param_index;
        binding.scalar_value.resize(rec->size);
        memcpy(binding.scalar_value.data(), rec->value, rec->size);

    } else {
        return false;  // Unknown param type
    }

    return true;
}

// Parse operations section (version 2 format using APICOpHeader with total_size)
static bool apic_parse_operations(const uint8_t* data, size_t size, APICGraphInternal* graph)
{
    if (!data || size < 4)
        return false;

    const uint8_t* ptr = data;
    const uint8_t* end = data + size;

    uint32_t num_ops = apic_read_value<uint32_t>(ptr);

    for (uint32_t i = 0; i < num_ops && ptr < end; i++) {
        // Read the operation header
        if (ptr + sizeof(APICOpHeader) > end)
            return false;

        const APICOpHeader* header = reinterpret_cast<const APICOpHeader*>(ptr);
        const uint8_t* op_start = ptr;
        uint8_t op_type = header->op_type;

        if (op_type == APIC_OP_KERNEL_LAUNCH) {
            // Read APICLaunchRecord header
            if (ptr + sizeof(APICLaunchRecord) > end)
                return false;

            const APICLaunchRecord* rec = reinterpret_cast<const APICLaunchRecord*>(ptr);
            const uint8_t* var_data = ptr + sizeof(APICLaunchRecord);

            APICKernelLaunch launch;
            launch.dim = rec->dim;
            launch.max_blocks = rec->max_blocks;
            launch.block_dim = rec->block_dim;
            launch.smem_bytes = rec->smem_bytes;
            launch.is_forward = rec->is_forward != 0;

            // Read variable-length strings
            launch.kernel_key = std::string(reinterpret_cast<const char*>(var_data), rec->kernel_key_len);
            var_data += rec->kernel_key_len;
            launch.module_hash = std::string(reinterpret_cast<const char*>(var_data), rec->module_hash_len);
            var_data += rec->module_hash_len;

            // Read parameter bindings
            const uint8_t* params_ptr = var_data;
            launch.param_bindings.resize(rec->num_params);
            for (uint16_t j = 0; j < rec->num_params; j++) {
                if (!apic_parse_param_binding(params_ptr, end, launch.param_bindings[j])) {
                    return false;
                }
            }

            // Get kernel name from metadata
            auto it = graph->kernels.find(launch.kernel_key);
            if (it != graph->kernels.end()) {
                launch.kernel_name = launch.is_forward ? it->second.forward_name : it->second.backward_name;
            }

            launch.kernel_func = nullptr;

            graph->launches.push_back(std::move(launch));

            APICOperationEntry entry;
            entry.is_launch = true;
            entry.index = graph->launches.size() - 1;
            graph->operations.push_back(entry);

        } else if (op_type == APIC_OP_MEMCPY_H2D) {
            // Read APICMemcpyH2DRecord
            if (ptr + sizeof(APICMemcpyH2DRecord) > end)
                return false;

            const APICMemcpyH2DRecord* rec = reinterpret_cast<const APICMemcpyH2DRecord*>(ptr);
            const uint8_t* data_ptr = ptr + sizeof(APICMemcpyH2DRecord);

            APICMemoryOp op;
            op.type = APIC_OP_MEMCPY_H2D;
            op.dst_region_id = rec->dst_region_id;
            op.dst_offset = rec->dst_offset;
            op.size = rec->size;
            op.src_data.resize(op.size);
            memcpy(op.src_data.data(), data_ptr, op.size);
            op.dst_ptr = nullptr;
            op.src_ptr = nullptr;

            graph->memory_ops.push_back(std::move(op));

            APICOperationEntry entry;
            entry.is_launch = false;
            entry.index = graph->memory_ops.size() - 1;
            graph->operations.push_back(entry);

        } else if (op_type == APIC_OP_MEMCPY_D2D) {
            // Read APICMemcpyD2DRecord
            if (ptr + sizeof(APICMemcpyD2DRecord) > end)
                return false;

            const APICMemcpyD2DRecord* rec = reinterpret_cast<const APICMemcpyD2DRecord*>(ptr);

            APICMemoryOp op;
            op.type = APIC_OP_MEMCPY_D2D;
            op.dst_region_id = rec->dst_region_id;
            op.src_region_id = rec->src_region_id;
            op.dst_offset = rec->dst_offset;
            op.src_offset = rec->src_offset;
            op.size = rec->size;
            op.dst_ptr = nullptr;
            op.src_ptr = nullptr;

            graph->memory_ops.push_back(std::move(op));

            APICOperationEntry entry;
            entry.is_launch = false;
            entry.index = graph->memory_ops.size() - 1;
            graph->operations.push_back(entry);

        } else if (op_type == APIC_OP_MEMSET) {
            // Read APICMemsetRecord
            if (ptr + sizeof(APICMemsetRecord) > end)
                return false;

            const APICMemsetRecord* rec = reinterpret_cast<const APICMemsetRecord*>(ptr);

            APICMemoryOp op;
            op.type = APIC_OP_MEMSET;
            op.dst_region_id = rec->region_id;
            op.dst_offset = rec->offset;
            op.value = rec->value;
            op.size = rec->size;
            op.dst_ptr = nullptr;

            graph->memory_ops.push_back(std::move(op));

            APICOperationEntry entry;
            entry.is_launch = false;
            entry.index = graph->memory_ops.size() - 1;
            graph->operations.push_back(entry);

        } else if (op_type == APIC_OP_ALLOC) {
            // Read APICAllocRecord - allocations are handled via memory_regions metadata
            // Just skip for now
        }

        // Advance to next operation using total_size
        ptr = op_start + header->total_size;
    }

    return true;
}

// Initialize memory regions with saved data (version 2 format using APICMemoryRegionRecord)
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

// Helper to create launch bounds for APIC replay
static apic_launch_bounds_t apic_make_launch_bounds(uint64_t dim)
{
    apic_launch_bounds_t bounds;
    bounds.ndim = 1;
    bounds.size = dim;
    bounds.shape[0] = (int)dim;
    for (int i = 1; i < APIC_LAUNCH_MAX_DIMS; i++) {
        bounds.shape[i] = 1;
    }
    return bounds;
}

// Resolve all pointers based on current region allocations
static void apic_resolve_pointers(APICGraphInternal* graph)
{
    // Resolve launch param bindings
    for (auto& launch : graph->launches) {
        for (auto& binding : launch.param_bindings) {
            if (binding.type == 1 && binding.region_id >= 0) {  // ARRAY
                auto it = graph->regions.find(binding.region_id);
                if (it != graph->regions.end() && it->second.ptr) {
                    binding.resolved_ptr = (void*)((uint8_t*)it->second.ptr + binding.byte_offset);
                }
            }
        }
    }

    // Resolve memory op pointers
    for (auto& op : graph->memory_ops) {
        if (op.dst_region_id >= 0) {
            auto it = graph->regions.find(op.dst_region_id);
            if (it != graph->regions.end() && it->second.ptr) {
                op.dst_ptr = (void*)((uint8_t*)it->second.ptr + op.dst_offset);
            }
        }
        if (op.type == 4 && op.src_region_id >= 0) {  // D2D
            auto it = graph->regions.find(op.src_region_id);
            if (it != graph->regions.end() && it->second.ptr) {
                op.src_ptr = (void*)((uint8_t*)it->second.ptr + op.src_offset);
            }
        }
    }
}

// Rebuild CUDA graph by replaying operations
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

    // Replay all operations
    for (const auto& entry : graph->operations) {
        if (entry.is_launch) {
            const APICKernelLaunch& launch = graph->launches[entry.index];

            // Get kernel function if not cached
            CUfunction kernel = launch.kernel_func;
            if (!kernel) {
                auto mod_it = graph->modules.find(launch.module_hash);
                if (mod_it == graph->modules.end() || !mod_it->second.cuda_module) {
                    wp::set_error_string("Module not loaded: %s", launch.module_hash.c_str());
                    success = false;
                    break;
                }

                err = cuModuleGetFunction_f(&kernel, mod_it->second.cuda_module, launch.kernel_name.c_str());
                if (err != CUDA_SUCCESS) {
                    wp::set_error_string("Failed to get kernel function %s: %d", launch.kernel_name.c_str(), err);
                    success = false;
                    break;
                }
                // Cache it (const_cast since we're caching)
                const_cast<APICKernelLaunch&>(launch).kernel_func = kernel;
            }

            // Build launch bounds and arguments
            // First arg is always launch_bounds_t
            apic_launch_bounds_t bounds = apic_make_launch_bounds(launch.dim);

            std::vector<void*> args;
            std::vector<std::unique_ptr<uint8_t[]>> arg_storage;

            args.push_back(&bounds);

            for (const auto& binding : launch.param_bindings) {
                if (binding.type == 1) {  // ARRAY
                    // Create array_t structure
                    auto arr = std::make_unique<uint8_t[]>(sizeof(apic_array_t));
                    apic_array_t* arr_ptr = reinterpret_cast<apic_array_t*>(arr.get());
                    memset(arr_ptr, 0, sizeof(apic_array_t));

                    arr_ptr->data = (uint64_t)binding.resolved_ptr;
                    arr_ptr->grad = 0;
                    arr_ptr->ndim = binding.ndim;
                    for (int d = 0; d < binding.ndim && d < APIC_MAX_DIMS; d++) {
                        arr_ptr->shape[d] = (int)binding.shape[d];
                        arr_ptr->strides[d] = (int)binding.strides[d];
                    }

                    args.push_back(arr_ptr);
                    arg_storage.push_back(std::move(arr));
                } else {  // SCALAR
                    auto scalar = std::make_unique<uint8_t[]>(binding.scalar_value.size());
                    memcpy(scalar.get(), binding.scalar_value.data(), binding.scalar_value.size());
                    args.push_back(scalar.get());
                    arg_storage.push_back(std::move(scalar));
                }
            }

            // Calculate grid dimensions
            int num_threads = launch.dim;
            int block_size = launch.block_dim;
            int max_blocks = launch.max_blocks;
            int num_blocks = (num_threads + block_size - 1) / block_size;
            if (max_blocks > 0 && num_blocks > max_blocks) {
                num_blocks = max_blocks;
            }

            // Launch kernel using wrapper function
            err = cuLaunchKernel_f(
                kernel, num_blocks, 1, 1,  // grid dim
                block_size, 1, 1,  // block dim
                launch.smem_bytes,  // shared mem
                stream,  // stream
                args.data(),  // kernel args
                nullptr  // extra
            );

            if (err != CUDA_SUCCESS) {
                wp::set_error_string("Failed to launch kernel %s: %d", launch.kernel_name.c_str(), err);
                success = false;
                break;
            }

        } else {
            const APICMemoryOp& op = graph->memory_ops[entry.index];

            if (op.type == 2) {  // H2D
                cuda_err = cudaMemcpyAsync(
                    op.dst_ptr, op.src_data.data(), op.size, cudaMemcpyHostToDevice, (cudaStream_t)stream
                );
            } else if (op.type == 4) {  // D2D
                cuda_err
                    = cudaMemcpyAsync(op.dst_ptr, op.src_ptr, op.size, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
            } else if (op.type == 5) {  // Memset
                cuda_err = cudaMemsetAsync(op.dst_ptr, op.value, op.size, (cudaStream_t)stream);
            }

            if (cuda_err != cudaSuccess) {
                wp::set_error_string("Failed memory operation: %d", cuda_err);
                success = false;
                break;
            }
        }
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
    graph->needs_rebuild = false;

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

    if (file_data.size() < WGF_HEADER_SIZE) {
        wp::set_error_string("Invalid WGF file: too small");
        return nullptr;
    }

    // Parse header
    const uint8_t* ptr = file_data.data();

    if (memcmp(ptr, WGF_MAGIC, 4) != 0) {
        wp::set_error_string("Invalid WGF file: bad magic");
        return nullptr;
    }
    ptr += 4;

    uint32_t version = apic_read_value<uint32_t>(ptr);
    if (version > WGF_VERSION) {
        wp::set_error_string("Unsupported WGF version: %u", version);
        return nullptr;
    }

    uint32_t flags = apic_read_value<uint32_t>(ptr);
    (void)flags;  // Reserved for future use

    uint32_t num_sections = apic_read_value<uint32_t>(ptr);
    uint64_t section_table_offset = apic_read_value<uint64_t>(ptr);
    uint32_t target_arch = apic_read_value<uint32_t>(ptr);

    // Create graph object
    APICGraphInternal* graph = new APICGraphInternal();
    graph->cuda_context = context;
    graph->target_arch = target_arch;
    graph->base_path = base_name;

    // Parse section table
    ptr = file_data.data() + section_table_offset;

    const uint8_t* metadata_ptr = nullptr;
    size_t metadata_size = 0;
    const uint8_t* memory_ptr = nullptr;
    size_t memory_size = 0;
    const uint8_t* operations_ptr = nullptr;
    size_t operations_size = 0;

    for (uint32_t i = 0; i < num_sections; i++) {
        uint32_t section_type = apic_read_value<uint32_t>(ptr);
        uint32_t section_flags = apic_read_value<uint32_t>(ptr);
        (void)section_flags;
        uint64_t section_offset = apic_read_value<uint64_t>(ptr);
        int64_t section_size = apic_read_value<int64_t>(ptr);
        int64_t uncompressed_size = apic_read_value<int64_t>(ptr);
        (void)uncompressed_size;  // TODO: compression support

        if (section_type == WGF_SECTION_METADATA) {
            metadata_ptr = file_data.data() + section_offset;
            metadata_size = section_size;
        } else if (section_type == WGF_SECTION_MEMORY) {
            memory_ptr = file_data.data() + section_offset;
            memory_size = section_size;
        } else if (section_type == WGF_SECTION_OPERATIONS) {
            operations_ptr = file_data.data() + section_offset;
            operations_size = section_size;
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

    // Resolve initial pointers
    apic_resolve_pointers(graph);

    return graph;
}

void wp_apic_destroy_graph(APICGraph graph)
{
    if (graph) {
        delete graph;
    }
}

int wp_apic_bind_input(APICGraph graph, const char* name, void* ptr, size_t size)
{
    if (!graph || !name)
        return 0;

    auto it = graph->input_bindings.find(name);
    if (it == graph->input_bindings.end()) {
        wp::set_error_string("Unknown input binding: %s", name);
        return 0;
    }

    uint32_t region_id = it->second;
    auto region_it = graph->regions.find(region_id);
    if (region_it == graph->regions.end()) {
        wp::set_error_string("Input region not found: %u", region_id);
        return 0;
    }

    if (size != region_it->second.size) {
        wp::set_error_string(
            "Size mismatch for input %s: expected %llu, got %llu", name, (unsigned long long)region_it->second.size,
            (unsigned long long)size
        );
        return 0;
    }

    region_it->second.ptr = ptr;
    region_it->second.external = true;
    graph->needs_rebuild = true;

    // Re-resolve pointers
    apic_resolve_pointers(graph);

    return 1;
}

int wp_apic_bind_output(APICGraph graph, const char* name, void* ptr, size_t size)
{
    if (!graph || !name)
        return 0;

    auto it = graph->output_bindings.find(name);
    if (it == graph->output_bindings.end()) {
        wp::set_error_string("Unknown output binding: %s", name);
        return 0;
    }

    uint32_t region_id = it->second;
    auto region_it = graph->regions.find(region_id);
    if (region_it == graph->regions.end()) {
        wp::set_error_string("Output region not found: %u", region_id);
        return 0;
    }

    if (size != region_it->second.size) {
        wp::set_error_string(
            "Size mismatch for output %s: expected %llu, got %llu", name, (unsigned long long)region_it->second.size,
            (unsigned long long)size
        );
        return 0;
    }

    region_it->second.ptr = ptr;
    region_it->second.external = true;
    graph->needs_rebuild = true;

    // Re-resolve pointers
    apic_resolve_pointers(graph);

    return 1;
}

void* wp_apic_get_cuda_graph(APICGraph graph)
{
    if (!graph)
        return nullptr;

    ContextGuard guard(graph->cuda_context);

    if (graph->needs_rebuild || !graph->cuda_graph) {
        // Get default stream using wrapper function
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

int wp_apic_launch_graph(APICGraph graph, void* stream)
{
    if (!graph)
        return 0;

    ContextGuard guard(graph->cuda_context);

    cudaGraphExec_t exec = (cudaGraphExec_t)wp_apic_get_cuda_graph_exec(graph);
    if (!exec)
        return 0;

    cudaError_t err = cudaGraphLaunch(exec, (cudaStream_t)stream);
    if (err != cudaSuccess) {
        wp::set_error_string("Failed to launch graph: %d", err);
        return 0;
    }

    return 1;
}

int wp_apic_get_num_inputs(APICGraph graph) { return graph ? (int)graph->input_names.size() : 0; }

int wp_apic_get_num_outputs(APICGraph graph) { return graph ? (int)graph->output_names.size() : 0; }

const char* wp_apic_get_input_name(APICGraph graph, int index)
{
    if (!graph || index < 0 || index >= (int)graph->input_names.size())
        return nullptr;
    return graph->input_names[index].c_str();
}

const char* wp_apic_get_output_name(APICGraph graph, int index)
{
    if (!graph || index < 0 || index >= (int)graph->output_names.size())
        return nullptr;
    return graph->output_names[index].c_str();
}

size_t wp_apic_get_input_size(APICGraph graph, const char* name)
{
    if (!graph || !name)
        return 0;
    auto it = graph->input_bindings.find(name);
    if (it == graph->input_bindings.end())
        return 0;
    auto region_it = graph->regions.find(it->second);
    if (region_it == graph->regions.end())
        return 0;
    return region_it->second.size;
}

size_t wp_apic_get_output_size(APICGraph graph, const char* name)
{
    if (!graph || !name)
        return 0;
    auto it = graph->output_bindings.find(name);
    if (it == graph->output_bindings.end())
        return 0;
    auto region_it = graph->regions.find(it->second);
    if (region_it == graph->regions.end())
        return 0;
    return region_it->second.size;
}
