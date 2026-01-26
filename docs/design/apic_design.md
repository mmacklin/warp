# APIC (API Capture) Design Document
## CUDA Graph Capture, Serialization, and Replay for Warp

### Version: 1.1
### Date: January 2026

---

## 1. Executive Summary

This document outlines the design for adding CUDA graph capture, serialization, and replay capabilities to Warp. The goal is to allow users to:

1. Capture a Warp computation graph including all kernel launches and memory operations
2. Serialize the captured graph to disk with all referenced memory and kernels
3. Load and execute the serialized graph later without requiring the original Python program
4. Generate a C++ header for embedding the captured computation into native applications

---

## 2. Requirements

### 2.1 Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | Capture kernel launches during `wp.capture_begin()` / `wp.capture_end()` | High |
| FR-2 | Capture memory operations (memcpy, memset, allocations) | High |
| FR-3 | Serialize captured graph to a custom binary format with `wp.capture_save()` | High |
| FR-4 | Deserialize and recreate graph with `wp.capture_load()` | High |
| FR-5 | Execute deserialized graph with `wp.capture_launch()` | High |
| FR-6 | Support `wp.capture_func(fn)` convenience API for capturing a callable | High |
| FR-7 | Serialize all referenced `wp.array` memory with proper aliasing handling | High |
| FR-8 | Serialize compiled CUDA kernels (CUBIN as separate files) | High |
| FR-9 | Generate C++ header for native application embedding | Medium |
| FR-10 | Support input/output array designation for graph parameters | High |
| FR-11 | Support `wp.Mesh`, `wp.Volume`, `wp.BVH` data structures | Medium |
| FR-12 | Handle array slicing/aliasing (same underlying memory) | High |

### 2.2 Non-Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| NF-1 | Minimize API changes - leverage existing infrastructure (Graph, Launch) | High |
| NF-2 | Seamless user experience | High |
| NF-3 | Backward compatibility with existing `wp.capture_*` APIs | High |
| NF-4 | Cross-platform support (Windows, Linux) | Medium |
| NF-5 | Version-tolerant binary format | Medium |
| NF-6 | Implementation split between Python and C++ as appropriate | Medium |

---

## 3. Current Architecture Analysis

### 3.1 Existing CUDA Graph Capture

**Location:** `warp/_src/context.py` (lines 7720-7827) and `warp/native/warp.cu` (lines 2698-2864)

Current flow:
```
wp.capture_begin(device, stream)
    -> wp_cuda_graph_begin_capture()
    -> cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)
    -> Creates CaptureInfo, registers in g_captures

[Kernel launches recorded by CUDA driver]

wp.capture_end()
    -> wp_cuda_graph_end_capture()
    -> cudaStreamEndCapture(stream, &graph)
    -> Returns Graph object with cudaGraph_t pointer

wp.capture_launch(graph)
    -> Lazy instantiation: cudaGraphInstantiateWithFlags()
    -> cudaGraphLaunch(graphExec, stream)
```

**Key Data Structures:**

```python
class Graph:
    device: Device              # Target CUDA device
    capture_id: int             # CUDA capture ID
    module_execs: set[ModuleExec]  # Retained module references
    graph_exec: ctypes.c_void_p    # cudaGraphExec_t (lazy)
    graph: ctypes.c_void_p         # cudaGraph_t
```

```cpp
struct CaptureInfo {
    CUstream stream;
    uint64_t id;
    bool external;
    std::vector<FreeInfo> tmp_allocs;
};
```

**Current Limitations:**
- No serialization support
- Graph objects are opaque CUDA pointers
- No metadata preservation
- Cannot introspect graph contents

### 3.2 Array and Memory System

**Location:** `warp/_src/types.py` (lines 2561+)

Key `array` attributes for serialization:
```python
array:
    ptr: int           # Raw device pointer
    dtype: type        # Data type
    ndim: int          # Number of dimensions
    shape: tuple       # Shape tuple
    strides: tuple     # Strides tuple
    device: Device     # CPU or CUDA device
    capacity: int      # Total allocated bytes
    is_contiguous: bool
    _ref: array | None # Parent array reference (for slices)
```

**Array Aliasing via Slicing:**
```python
a = wp.zeros(100, dtype=float)
b = a[10:50]  # b.ptr = a.ptr + 10*sizeof(float), b._ref = a
c = a[20:30]  # c.ptr = a.ptr + 20*sizeof(float), c._ref = a
# b and c share memory with a but have different views
```

The `_ref` chain allows tracking back to the original allocation.

### 3.3 Kernel and Module System

**Location:** `warp/_src/context.py` (lines 748+, 2217+, 2336+)

```python
class Kernel:
    func: Callable      # Original Python function
    module: Module      # Containing module
    key: str            # Fully qualified name
    adj: Adjoint        # AST and code generation info
    hash: str           # SHA256 hash (computed during build)

class Module:
    name: str           # Module name
    kernels: dict       # kernel.key -> Kernel
    execs: dict         # (device.context, block_dim) -> ModuleExec

class ModuleExec:
    handle: void*       # CUDA module handle
    module_hash: str    # Hash when loaded
    device: Device
    kernel_hooks: dict  # kernel.adj -> KernelHooks

class KernelHooks:
    forward: void*      # Forward kernel function pointer
    backward: void*     # Backward kernel function pointer
    forward_smem_bytes: int
    backward_smem_bytes: int
```

**Kernel Launch Parameters:**
```python
# From launch() function (context.py:6886+)
kernel_params = [
    bounds,           # launch_bounds_t (dim info)
    *input_args,      # Converted via pack_arg()
    *output_args,     # Converted via pack_arg()
    *adj_input_args,  # For backward pass
    *adj_output_args  # For backward pass
]

# array arguments become array_t structures:
class array_t(ctypes.Structure):
    _fields_ = (
        ("data", ctypes.c_uint64),      # Raw pointer
        ("grad", ctypes.c_uint64),      # Gradient pointer
        ("ndim", ctypes.c_int32),
        ("shape", ctypes.c_int32 * ARRAY_MAX_DIMS),
        ("strides", ctypes.c_int32 * ARRAY_MAX_DIMS),
    )
```

### 3.4 PTX/CUBIN Compilation and Loading

**Location:** `warp/_src/build.py` (lines 39-111)

```python
# Compilation: build_cuda() -> wp_cuda_compile_program()
# Produces: .ptx or .cubin files in kernel cache

# Loading: load_cuda() -> wp_cuda_load_module()
# Returns: CUmodule handle

# Kernel lookup: wp_cuda_get_kernel(module, name)
# Returns: CUfunction handle
```

Cached artifacts location: `{kernel_cache_dir}/wp_{module_name}_{hash[:7]}/`

---

## 4. Proposed Architecture

### 4.1 High-Level Design

```
                    +------------------+
                    |   User Code      |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
    +-------------------+         +-------------------+
    | wp.capture_func() |         | wp.capture_save() |
    +-------------------+         +-------------------+
              |                             |
              v                             v
    +-------------------+         +-------------------+
    | APICapture Layer  |-------->| WarpGraphFile     |
    | (Recording Mode)  |         | (.wgf binary)     |
    +-------------------+         +-------------------+
              |                             |
              |                             v
              v                   +-------------------+
    +-------------------+         | wp.capture_load() |
    | Captured State    |         +-------------------+
    | - Operations[]    |                   |
    | - Memory Regions  |                   v
    | - Module CUBINs   |         +-------------------+
    +-------------------+         | Graph (extended)  |
                                  | - CUDA Graph      |
                                  | - Memory Mapping  |
                                  +-------------------+
                                            |
                                            v
                                  +-------------------+
                                  |wp.capture_launch()|
                                  +-------------------+
```

### 4.2 Component Design

#### 4.2.1 APIC Layer (C++ and Python)

The APIC layer intercepts Warp operations during capture. The implementation is split:
- **C++ (warp.cu/warp.cpp)**: Low-level operation recording, memory tracking
- **Python (context.py)**: High-level orchestration, serialization

**C++ APIC State (warp.cu):**

```cpp
// APIC recording state
struct APICState {
    bool recording = false;
    std::vector<APICOperation> operations;
    std::unordered_map<uint64_t, APICMemoryRegion> memory_regions;
    std::unordered_map<std::string, APICKernelInfo> kernels;
};

struct APICMemoryRegion {
    uint32_t region_id;
    uint64_t base_ptr;       // Base allocation pointer
    uint64_t size;           // Size in bytes
    uint8_t role;            // input/output/internal
};

struct APICOperation {
    uint8_t op_type;         // APIC_OP_KERNEL, APIC_OP_MEMCPY, etc.
    // Union or variant for operation-specific data
};

// Thread-local APIC state
static thread_local APICState* g_apic_state = nullptr;

// APIC API
void wp_apic_begin(APICState* state);
void wp_apic_end();
void wp_apic_record_kernel(void* kernel, size_t dim, int max_blocks,
                           int block_dim, int smem_bytes, void** args);
void wp_apic_record_memcpy(void* dst, void* src, size_t size, int kind);
void wp_apic_record_memset(void* dst, int value, size_t size);
void wp_apic_track_memory(uint64_t ptr, uint64_t size, uint8_t role);
```

**Python APICapture Class:**

```python
class APICapture:
    """Records API calls during CUDA graph capture for serialization."""

    def __init__(self, device: Device, stream: Stream = None):
        self.device = device
        self.stream = stream or device.stream
        self.launches: list[Launch] = []  # Reuse existing Launch objects
        self.memory_ops: list[MemoryOp] = []
        self.memory_regions: dict[int, MemoryRegion] = {}
        self.kernels: dict[str, KernelInfo] = {}
        self.inputs: dict[str, ArrayBinding] = {}
        self.outputs: dict[str, ArrayBinding] = {}
        self._c_state = None  # Pointer to C++ APICState

    def begin(self):
        """Start APIC recording (calls into C++)."""
        self._c_state = runtime.core.wp_apic_create_state()
        runtime.core.wp_apic_begin(self._c_state)

    def end(self):
        """End APIC recording and collect results."""
        runtime.core.wp_apic_end()
        # Retrieve recorded operations from C++ state

    def record_launch(self, launch: Launch):
        """Record a kernel launch (reusing existing Launch object)."""
        self.launches.append(launch)
        # Also record kernel info if not seen before
        if launch.kernel.key not in self.kernels:
            self.kernels[launch.kernel.key] = self._extract_kernel_info(launch.kernel)
```

**Memory Operation Recording:**

```python
@dataclass
class MemoryOp:
    """Base class for memory operations."""
    pass

@dataclass
class MemcpyOp(MemoryOp):
    dst_region_id: int
    dst_offset: int
    src_region_id: int | None  # None for host source
    src_offset: int
    size: int
    kind: str  # "HtoD", "DtoH", "DtoD"

@dataclass
class MemsetOp(MemoryOp):
    region_id: int
    offset: int
    value: int
    size: int

@dataclass
class AllocOp(MemoryOp):
    region_id: int
    size: int
```

**Memory Region Tracking:**

```python
@dataclass
class MemoryRegion:
    """Represents a contiguous memory allocation."""
    region_id: int
    size: int                # Size in bytes
    device_ptr: int          # Original device pointer
    role: str                # "input", "output", "internal", "input_output"
    element_size: int        # Size of one element in bytes (for type reconstruction)
    initial_data: bytes | None  # For internal arrays: serialized content

    # Tracking info for finding the base allocation
    base_array_ptr: int      # The root array pointer (after resolving _ref chain)
```

#### 4.2.2 Module and Kernel Serialization

In Warp, a **Module** contains multiple kernels. We export one .cubin file per module (not per kernel), stored in a `_modules/` directory alongside the .wgf file.

**File Layout:**
```
my_graph.wgf                    # Main graph file
my_graph_modules/               # Module directory (one cubin per module)
    my_module_abc1234.cubin     # Contains all kernels from my_module
    other_module_def5678.cubin  # Contains all kernels from other_module
    ...
my_graph_memory/                # Optional: large memory blobs
    region_0.bin
```

**Module Tracking During Capture:**

The APICapture class tracks unique modules encountered during capture:

```python
class APICapture:
    def __init__(self, ...):
        ...
        self.modules: dict[str, ModuleInfo] = {}  # module_hash -> ModuleInfo
        self.kernels: dict[str, KernelInfo] = {}  # kernel_key -> KernelInfo

    def record_launch(self, launch: Launch):
        """Record a kernel launch, tracking its module."""
        self.launches.append(launch)

        kernel = launch.kernel
        module = kernel.module
        module_exec = launch.module_exec
        module_hash = module_exec.module_hash

        # Track unique modules
        if module_hash not in self.modules:
            self.modules[module_hash] = ModuleInfo(
                module_name=module.name,
                module_hash=module_hash,
                target_arch=self.device.arch,
                cubin_filename=f"{module.name}_{module_hash[:8]}.cubin"
            )

        # Track kernel info (references its module)
        if kernel.key not in self.kernels:
            hooks = module_exec.get_kernel_hooks(kernel)
            self.kernels[kernel.key] = KernelInfo(
                kernel_key=kernel.key,
                module_hash=module_hash,  # Reference to module
                forward_name=f"{kernel.key}_{module_hash[:8]}_cuda_kernel_forward",
                backward_name=f"{kernel.key}_{module_hash[:8]}_cuda_kernel_backward" if hooks.backward else None,
                forward_smem_bytes=hooks.forward_smem_bytes,
                backward_smem_bytes=hooks.backward_smem_bytes if hooks.backward else 0,
                block_dim=launch.block_dim
            )
```

**Data Structures:**

```python
@dataclass
class ModuleInfo:
    """Module metadata - one cubin file contains all kernels from a module."""
    module_name: str
    module_hash: str             # Hash identifying the compiled module
    target_arch: int             # SM version (e.g., 86)
    cubin_filename: str          # Relative path to .cubin file

@dataclass
class KernelInfo:
    """Kernel metadata - references its containing module."""
    kernel_key: str
    module_hash: str             # Reference to ModuleInfo
    forward_name: str            # Mangled kernel function name
    backward_name: str | None
    forward_smem_bytes: int
    backward_smem_bytes: int
    block_dim: int
```

**Module Export:**

```python
def export_modules(self, output_dir: str):
    """Export all unique modules used in the captured graph."""
    modules_dir = os.path.join(output_dir, f"{self.graph_name}_modules")
    os.makedirs(modules_dir, exist_ok=True)

    for module_hash, info in self.modules.items():
        # Find the module and its cached cubin
        module = self._find_module_by_hash(module_hash)
        module_exec = module.load(self.device)

        # Source: cached .cubin file
        cubin_path = module._get_compile_output_name(
            self.device, module_hash, ".cubin"
        )

        # If no cubin exists, generate from PTX
        if not os.path.exists(cubin_path):
            ptx_path = module._get_compile_output_name(
                self.device, module_hash, ".ptx"
            )
            runtime.core.wp_cuda_compile_ptx_to_cubin(
                ptx_path.encode(), cubin_path.encode(), self.device.arch
            )

        # Copy to output directory
        dest_path = os.path.join(modules_dir, info.cubin_filename)
        shutil.copy2(cubin_path, dest_path)
```

#### 4.2.3 Binary File Format (.wgf - Warp Graph File)

The .wgf file contains metadata and operations. CUBIN files are stored separately.

**Directory Structure:**
```
output_dir/
    my_graph.wgf                # Main graph file (metadata + operations)
    my_graph_modules/           # Module directory (one cubin per Warp module)
        my_module_abc123.cubin  # Standard CUDA binary format
        other_module_def456.cubin
    my_graph_memory/            # Initial memory data (optional, for large data)
        region_0.bin
        region_1.bin
```

**File Structure:**

```
+-------------------+
| Header (64 bytes) |
+-------------------+
| Section Table     |
+-------------------+
| Metadata Section  |  (JSON: kernel info, bindings)
+-------------------+
| Memory Section    |  (inline or references to external files)
+-------------------+
| Operations Section|
+-------------------+
```

**Header Format:**

```c
struct WGFHeader {
    char magic[4];           // "WGF1"
    uint32_t version;        // Format version
    uint32_t flags;          // Feature flags
    uint32_t num_sections;   // Number of sections
    uint64_t section_table_offset;
    uint32_t target_arch;    // CUDA SM version
    uint32_t reserved[10];   // Future use
};

struct SectionEntry {
    uint32_t type;           // Section type enum
    uint32_t flags;          // Section-specific flags
    uint64_t offset;         // Offset from file start
    uint64_t size;           // Section size in bytes
    uint64_t uncompressed_size;  // If compressed
};
```

**Section Types:**

| Type | ID | Description |
|------|-----|-------------|
| METADATA | 0x01 | JSON metadata (kernel refs, bindings, array info) |
| MEMORY | 0x02 | Initial memory contents (or external file refs) |
| OPERATIONS | 0x03 | Serialized operation sequence |

**Operations Encoding:**

```c
// Operation types
enum APICOpType {
    APIC_OP_KERNEL_LAUNCH = 1,
    APIC_OP_MEMCPY_HtoD = 2,
    APIC_OP_MEMCPY_DtoD = 3,
    APIC_OP_MEMSET = 4,
};

// Operation header
struct OpHeader {
    uint8_t op_type;
    uint8_t flags;
    uint16_t num_params;
    uint32_t data_size;      // Size of following data
};

// Kernel launch operation
struct KernelLaunchData {
    uint32_t kernel_index;   // Index into kernels metadata
    uint64_t dim;            // Launch dimension
    int32_t max_blocks;
    int32_t block_dim;
    int32_t smem_bytes;
    uint8_t is_forward;
    // Followed by: ParamBinding[num_params]
};

// Parameter binding - simplified for arrays
struct ParamBinding {
    uint8_t param_type;      // PARAM_ARRAY, PARAM_SCALAR
    uint8_t flags;
    uint16_t reserved;
    uint32_t region_id;      // Memory region reference
    uint64_t offset;         // Byte offset within region
    int32_t shape[ARRAY_MAX_DIMS];
    int32_t strides[ARRAY_MAX_DIMS];
    int32_t ndim;
    uint32_t element_size;   // Size of one element in bytes (for type-agnostic handling)
    // For scalars: followed by value bytes (size = element_size)
};

// Memory operations
struct MemcpyData {
    uint32_t dst_region_id;
    uint64_t dst_offset;
    uint32_t src_region_id;  // 0xFFFFFFFF = host source (data follows)
    uint64_t src_offset;
    uint64_t size;
};

struct MemsetData {
    uint32_t region_id;
    uint64_t offset;
    uint64_t size;
    int32_t value;
};
```

**Array Data Encoding:**

Arrays are stored with minimal type information - just the byte size per element:

```c
struct ArrayRegionInfo {
    uint32_t region_id;
    uint64_t size_bytes;         // Total allocation size
    uint32_t element_size;       // Bytes per element (e.g., 4 for float, 32 for mat22f)
    uint8_t role;                // INPUT, OUTPUT, INTERNAL, INPUT_OUTPUT
    uint8_t storage;             // INLINE or EXTERNAL_FILE
    // If INLINE: followed by size_bytes of data
    // If EXTERNAL: followed by null-terminated filename
};
```

This approach handles arbitrary vector/matrix types (e.g., `vec(8, float16)` = 16 bytes) without deep type reflection.

#### 4.2.4 Extending the Graph Class

Rather than creating a new `LoadedGraph` class, we extend the existing `Graph` class to support serialization and loading. This maintains API consistency and allows seamless use of loaded graphs with existing code.

**Extended Graph Class:**

```python
class Graph:
    """Warp CUDA graph - extended to support serialization."""

    def __init__(self, device: Device, capture_id: int = None):
        self.device = device
        self.capture_id = capture_id
        self.module_execs: set[ModuleExec] = set()
        self.graph_exec: ctypes.c_void_p | None = None
        self.graph: ctypes.c_void_p | None = None

        # APIC extensions (populated when loaded from file)
        self._apic_state: APICapture | None = None
        self._loaded_modules: dict[str, ctypes.c_void_p] = {}  # module_hash -> handle
        self._loaded_kernels: dict[str, LoadedKernel] = {}     # kernel_key -> LoadedKernel
        self._memory_regions: dict[int, DeviceAllocation] = {}
        self._input_bindings: dict[str, InputBinding] = {}
        self._output_bindings: dict[str, OutputBinding] = {}
        self._source_path: str | None = None  # Path if loaded from file

    @classmethod
    def load(cls, path: str, device: Device = None) -> "Graph":
        """Load a graph from a .wgf file."""
        device = device or wp.get_device()
        graph = cls(device)
        graph._source_path = path
        graph._load_from_file(path)
        return graph

    def save(self, path: str,
             inputs: dict[str, array] = None,
             outputs: dict[str, array] = None,
             generate_cpp_header: bool = False) -> None:
        """Save this graph to a .wgf file."""
        if self._apic_state is None:
            raise RuntimeError("Graph was not captured with APIC enabled")
        self._save_to_file(path, inputs, outputs, generate_cpp_header)

    def _load_from_file(self, path: str):
        """Load and parse the .wgf file, reconstruct CUDA graph."""
        # Parse header and sections
        # Load module CUBIN files from modules directory
        # Get kernel function handles from loaded modules
        # Allocate memory regions
        # Reconstruct CUDA graph via capture replay

    def _allocate_memory(self):
        """Allocate device memory for all regions."""

    def _load_modules(self, modules_dir: str):
        """Load CUBIN files for all modules."""

    def _load_kernels(self):
        """Get kernel function handles from loaded modules."""

    def _build_cuda_graph(self):
        """Construct the CUDA graph by replaying recorded operations."""

    def bind_input(self, name: str, arr: array) -> None:
        """Bind an input array (updates graph exec parameters)."""
        if name not in self._input_bindings:
            raise KeyError(f"Unknown input binding: {name}")
        binding = self._input_bindings[name]
        self._update_binding(binding, arr)

    def bind_output(self, name: str, arr: array) -> None:
        """Bind an output array (updates graph exec parameters)."""
        if name not in self._output_bindings:
            raise KeyError(f"Unknown output binding: {name}")
        binding = self._output_bindings[name]
        self._update_binding(binding, arr)

    def _update_binding(self, binding: ArrayBinding, arr: array):
        """Update kernel node parameters for a binding."""
        # Validate array properties
        if binding.element_size != arr.dtype._length_ * arr.dtype._type_._length_:
            raise TypeError(f"Element size mismatch")
        # Use cudaGraphExecKernelNodeSetParams to update
        for node_ref in binding.node_references:
            runtime.core.wp_apic_update_kernel_node(
                self.graph_exec, node_ref.node_id,
                node_ref.param_index, arr.ptr
            )

    @property
    def inputs(self) -> dict[str, InputBinding]:
        """Get input binding information."""
        return self._input_bindings

    @property
    def outputs(self) -> dict[str, OutputBinding]:
        """Get output binding information."""
        return self._output_bindings
```

**Graph Reconstruction via Replay:**

Rather than deserializing a CUDA graph directly (not supported by CUDA), we reconstruct it by replaying operations:

```python
def _build_cuda_graph(self):
    """Reconstruct CUDA graph from recorded operations."""

    # Start capture on our device/stream
    wp.capture_begin(self.device, self.stream)

    try:
        for op in self._apic_state.operations:
            if isinstance(op, Launch):
                # Replay kernel launch
                self._replay_kernel_launch(op)
            elif isinstance(op, MemcpyOp):
                # Replay memory copy
                self._replay_memcpy(op)
            elif isinstance(op, MemsetOp):
                # Replay memory set
                self._replay_memset(op)

        # End capture - this populates self.graph
        captured = wp.capture_end()
        self.graph = captured.graph
        self.graph_exec = None  # Will be created on first launch

    except Exception:
        wp.capture_end()  # Clean up capture state
        raise
```

#### 4.2.5 Input/Output Binding System

To support parameterized graphs, we need a binding system:

```python
@dataclass
class InputBinding:
    """Describes an input array slot."""
    name: str                # User-visible name
    index: int               # Binding index
    dtype: type              # Expected data type
    shape: tuple | None      # Expected shape (None = any)
    region_id: int           # Which memory region this replaces

@dataclass
class OutputBinding:
    """Describes an output array slot."""
    name: str
    index: int
    dtype: type
    shape: tuple | None
    region_id: int
```

**Graph Node Update for Bindings:**

When a user binds an array, we need to update the kernel arguments in the CUDA graph:

```python
def bind_input(self, name: str, arr: array):
    binding = self._find_binding(name, self._input_bindings)

    # Validate array properties
    if binding.dtype != arr.dtype:
        raise TypeError(f"Expected dtype {binding.dtype}, got {arr.dtype}")
    if binding.shape and arr.shape != binding.shape:
        raise ValueError(f"Expected shape {binding.shape}, got {arr.shape}")

    # Update CUDA graph kernel node parameters
    # This uses cudaGraphExecKernelNodeSetParams
    for node_id, param_idx in binding.node_references:
        self._update_kernel_node_param(node_id, param_idx, arr)
```

#### 4.2.6 C++ Header Generation

Generate a self-contained C++ header for native embedding. The header references the .wgf file and module .cubin files (in `_modules/` directory), providing a clean C++ interface.

```cpp
// Generated file: my_computation.h

#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <string>

namespace warp {

// Array descriptor matching Warp's array_t
struct array_t {
    uint64_t data;
    uint64_t grad;
    int32_t ndim;
    int32_t shape[4];
    int32_t strides[4];
};

// Input/output binding info
struct Binding {
    const char* name;
    uint32_t element_size;  // Bytes per element
    int32_t ndim;
    int32_t shape[4];       // -1 = any dimension
};

// Generated graph class
class MyComputation {
public:
    // Constructor - loads .wgf and module .cubin files from _modules/ directory
    explicit MyComputation(const char* base_path = "my_computation");
    ~MyComputation();

    // Query interface
    int num_inputs() const;
    int num_outputs() const;
    const Binding& input_binding(int index) const;
    const Binding& input_binding(const char* name) const;
    const Binding& output_binding(int index) const;
    const Binding& output_binding(const char* name) const;

    // Bind arrays - validates element_size matches
    void bind_input(int index, void* ptr, const int32_t* shape,
                    const int32_t* strides, int ndim);
    void bind_input(const char* name, void* ptr, const int32_t* shape,
                    const int32_t* strides, int ndim);
    void bind_output(int index, void* ptr, const int32_t* shape,
                     const int32_t* strides, int ndim);
    void bind_output(const char* name, void* ptr, const int32_t* shape,
                     const int32_t* strides, int ndim);

    // Convenience: bind from array_t
    void bind_input(int index, const array_t& arr);
    void bind_input(const char* name, const array_t& arr);
    void bind_output(int index, const array_t& arr);
    void bind_output(const char* name, const array_t& arr);

    // Execute on stream (nullptr = default stream)
    void launch(cudaStream_t stream = nullptr);

private:
    struct Impl;
    Impl* impl_;
};

} // namespace warp
```

**Implementation File (my_computation.cpp):**

The implementation is generated alongside the header and handles:
- Loading the .wgf file and parsing metadata
- Loading module .cubin files from `_modules/` directory
- Getting kernel function handles from loaded modules
- Reconstructing the CUDA graph via capture replay
- Implementing binding updates via `cudaGraphExecKernelNodeSetParams`

```cpp
// Generated implementation
#include "my_computation.h"
#include <fstream>
#include <vector>
#include <unordered_map>

namespace warp {

struct MyComputation::Impl {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    // Loaded modules (one cubin = one module, contains multiple kernels)
    std::unordered_map<std::string, CUmodule> modules;  // module_hash -> handle

    // Kernel function pointers (looked up from modules)
    struct KernelHandles {
        CUfunction forward;
        CUfunction backward;
        int smem_forward;
        int smem_backward;
        int block_dim;
    };
    std::unordered_map<std::string, KernelHandles> kernels;  // kernel_key -> handles

    std::vector<Binding> inputs;
    std::vector<Binding> outputs;
    std::unordered_map<std::string, int> input_names;
    std::unordered_map<std::string, int> output_names;

    // Node references for binding updates
    struct NodeRef {
        cudaGraphNode_t node;
        int param_index;
    };
    std::vector<std::vector<NodeRef>> input_nodes;
    std::vector<std::vector<NodeRef>> output_nodes;

    void load(const char* base_path);
    void load_modules(const char* modules_dir);
    void load_kernels();
    void build_graph();
};

// ... implementation details ...

} // namespace warp
```

---

## 5. API Design

### 5.1 New Public APIs

```python
# Core capture/save/load APIs
def capture_save(graph: Graph, path: str,
                 inputs: dict[str, array] = None,
                 outputs: dict[str, array] = None,
                 generate_cpp_header: bool = False) -> None:
    """
    Save a captured CUDA graph to a .wgf file.

    Args:
        graph: A graph obtained from wp.capture_end() with apic=True
        path: Output file path (.wgf extension)
        inputs: Named arrays to mark as inputs (updateable at load time)
        outputs: Named arrays to mark as outputs (updateable at load time)
        generate_cpp_header: If True, also generate a .h file

    Creates:
        path.wgf                - Main graph file
        path_modules/           - Directory with .cubin files (one per Warp module)
        path_memory/            - Directory with large memory blobs (optional)
        path.h                  - C++ header (if generate_cpp_header=True)
    """

def capture_load(path: str, device: Device = None) -> Graph:
    """
    Load a serialized graph from a .wgf file.

    Args:
        path: Path to .wgf file
        device: Target device (default: current CUDA device)

    Returns:
        Graph object ready for binding and execution (same type as capture_end())
    """

def capture_func(fn: Callable,
                 inputs: dict[str, array],
                 outputs: dict[str, array],
                 path: str = None,
                 device: Device = None) -> Graph | None:
    """
    Convenience function to capture, and optionally save, a computation.

    Args:
        fn: A callable that takes no arguments and performs Warp operations
        inputs: Named input arrays used by fn
        outputs: Named output arrays produced by fn
        path: If provided, save the captured graph to this path
        device: Target device

    Returns:
        Graph object if path is None, else None (graph is saved to file)

    Example:
        def compute():
            wp.launch(my_kernel, dim=N, inputs=[a], outputs=[b])

        wp.capture_func(compute,
                        inputs={"positions": a},
                        outputs={"results": b},
                        path="my_computation.wgf")
    """
```

### 5.2 Extended capture_begin/capture_end

To enable APIC recording, we add an `apic` parameter:

```python
def capture_begin(device: Device = None,
                  stream: Stream = None,
                  force_module_load: bool = None,
                  external: bool = False,
                  apic: bool = False) -> None:
    """
    Begin CUDA graph capture.

    Args:
        device: Target device
        stream: Stream to capture on
        force_module_load: Force loading all modules before capture
        external: Whether capture was started externally
        apic: Enable API capture for serialization support
    """

def capture_end(device: Device = None, stream: Stream = None) -> Graph:
    """
    End CUDA graph capture.

    Returns:
        Graph object. If apic=True was set in capture_begin,
        the graph will have serialization support (save/bind methods).
    """
```

### 5.3 Extended Graph Class API

The existing `Graph` class gains new methods:

```python
class Graph:
    # Existing methods (unchanged)
    def __init__(self, device: Device, capture_id: int): ...
    def retain_module_exec(self, module_exec: ModuleExec): ...

    # New class method for loading
    @classmethod
    def load(cls, path: str, device: Device = None) -> "Graph":
        """Load a graph from a .wgf file."""

    # New instance methods for APIC-enabled graphs
    def save(self, path: str,
             inputs: dict[str, array] = None,
             outputs: dict[str, array] = None,
             generate_cpp_header: bool = False) -> None:
        """Save this graph to a .wgf file (requires apic=True during capture)."""

    def bind_input(self, name: str, arr: array) -> None:
        """Bind an array to a named input slot."""

    def bind_output(self, name: str, arr: array) -> None:
        """Bind an array to a named output slot."""

    @property
    def inputs(self) -> dict[str, InputBinding]:
        """Get input binding information (empty if not loaded from file)."""

    @property
    def outputs(self) -> dict[str, OutputBinding]:
        """Get output binding information (empty if not loaded from file)."""

    @property
    def is_serializable(self) -> bool:
        """True if graph was captured with apic=True or loaded from file."""
```

### 5.4 capture_launch (Unchanged)

The existing `capture_launch` works with all Graph objects:

```python
def capture_launch(graph: Graph, stream: Stream = None) -> None:
    """Launch a captured CUDA graph."""
    # Existing implementation - works for both regular and loaded graphs
```

---

## 6. Implementation Strategy

### 6.1 Phase 1: C++ APIC Infrastructure

1. **APICState structure in warp.cu** - Core recording state
2. **Hook wp_cuda_launch_kernel()** - Record kernel launches when APIC active
3. **Hook memory operations** - Record memcpy, memset, alloc
4. **wp_apic_begin/end()** - Start/stop recording
5. **wp_apic_get_operations()** - Retrieve recorded ops for Python

### 6.2 Phase 2: Python APIC Layer (Arrays Only)

1. **Extend Graph class** - Add APIC fields and methods
2. **APICapture class** - Python-side orchestration
3. **Memory region tracking** - Handle array aliasing via `_ref` chain
4. **Launch recording** - Capture `Launch` objects with full context
5. **Basic .wgf file format** - Header, sections, JSON metadata
6. **capture_save()** - Serialize to .wgf + copy .cubin files
7. **Graph.load()** - Load .wgf and reconstruct graph

### 6.3 Phase 3: Input/Output Bindings

1. **Binding specification** - Mark arrays as inputs/outputs in save()
2. **Track graph nodes** - Record which nodes use which bindings
3. **Graph node update** - Implement `cudaGraphExecKernelNodeSetParams` wrapper
4. **Validation** - Element size checking for bound arrays
5. **Graph.bind_input/output()** - Full binding API

### 6.4 Phase 4: C++ Header Generation

1. **Template system** - Generate C++ header + implementation
2. **Standalone loader** - C++ code to load .wgf without Python
3. **Testing** - Verify C++ integration compiles and runs

### 6.5 Phase 5: Complex Data Structures

1. **wp.Mesh** - Serialize points, indices arrays (uses Phase 2 array support)
2. **wp.BVH** - Serialize bounds arrays
3. **wp.Volume** - Serialize NanoVDB data as uint8 array
4. **Struct arrays** - Already handled via element_size approach

---

## 7. Integration Points

### 7.1 Modifications to Existing Code

**warp/native/warp.cu (C++ Layer):**
- Add APIC state management (`APICState`, `g_apic_state`)
- Modify `wp_cuda_launch_kernel()` to record when APIC active
- Add memory operation hooks:
  - `wp_alloc_device_*` - record allocations
  - `wp_memcpy_*` - record memory copies
  - `wp_memset_device` - record memory sets
- Add APIC-specific functions:
  - `wp_apic_create_state()` / `wp_apic_destroy_state()`
  - `wp_apic_begin()` / `wp_apic_end()`
  - `wp_apic_get_operations()` - retrieve recorded ops
  - `wp_apic_update_kernel_node()` - update graph exec params
  - `wp_cuda_load_module_from_data()` - load CUBIN from memory

**warp/native/warp.h:**
- Export new APIC functions
- Define APIC data structures for C/Python interop

**warp/_src/context.py (Python Layer):**
- Extend `Graph` class with APIC fields and methods
- Add `apic` parameter to `capture_begin()`
- Modify `launch()` to record `Launch` objects when APIC active
- Add public APIs: `capture_save`, `capture_load`, `capture_func`
- Add thread-local `_apic_capture` state

**warp/_src/types.py:**
- Add `get_element_size()` helper for arrays
- Ensure `_ref` chain is properly serializable

**warp/_src/build.py:**
- Add `get_kernel_cubin_path()` helper
- Add `compile_ptx_to_cubin()` if needed

### 7.2 New Files

```
warp/_src/apic/
    __init__.py           # Public exports
    capture.py            # APICapture class (Python side)
    serialize.py          # Serialization/deserialization
    format.py             # .wgf file format handling
    bindings.py           # Input/output binding system
    cpp_gen.py            # C++ header generation

warp/native/
    apic.h                # APIC C++ declarations
    apic.cpp              # APIC C++ implementation (compiled into warp.dll)
```

### 7.3 Memory Operation Hooks

The C++ layer needs to intercept memory operations during APIC capture:

```cpp
// In warp.cu - modified allocation function
void* wp_alloc_device_default(void* context, size_t size) {
    ContextGuard guard(context);
    void* ptr;
    check_cuda(cudaMalloc(&ptr, size));

    // APIC hook
    if (g_apic_state && g_apic_state->recording) {
        g_apic_state->record_alloc(ptr, size);
    }

    return ptr;
}

// Modified memcpy
void wp_memcpy_d2d(void* context, void* dst, void* src, size_t size) {
    ContextGuard guard(context);

    // APIC hook
    if (g_apic_state && g_apic_state->recording) {
        g_apic_state->record_memcpy(dst, src, size, APIC_MEMCPY_D2D);
    }

    check_cuda(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice,
                               get_current_stream()));
}

// Modified kernel launch
size_t wp_cuda_launch_kernel(void* context, void* kernel, size_t dim,
                             int max_blocks, int block_dim,
                             int shared_memory_bytes, void** args,
                             void* stream) {
    ContextGuard guard(context);

    // APIC hook
    if (g_apic_state && g_apic_state->recording) {
        g_apic_state->record_kernel_launch(
            kernel, dim, max_blocks, block_dim, shared_memory_bytes, args
        );
    }

    // ... existing launch code ...
}
```

### 7.4 Python-C++ Interface

The Python side retrieves recorded operations from C++:

```python
# In context.py
def capture_end(device=None, stream=None):
    # ... existing code ...

    graph = runtime.captures[stream].get(capture_id)

    # If APIC was enabled, finalize recording
    if graph._apic_state is not None:
        # Get recorded operations from C++ layer
        ops_data = runtime.core.wp_apic_get_operations(graph._apic_state._c_state)
        graph._apic_state.parse_operations(ops_data)

        # Collect Launch objects recorded on Python side
        # (these have richer type information than C++ layer)

    # ... rest of existing code ...
```

---

## 8. Memory Aliasing Handling

### 8.1 Problem Statement

When arrays share memory (via slicing), we must:
1. Only serialize the base allocation once
2. Track offsets for each aliased view
3. Correctly reconstruct views on load

### 8.2 Solution: Region Tracking

```python
def track_array(self, arr: array, role: str = "internal"):
    """Track an array, resolving to its base allocation."""

    # Walk the _ref chain to find base allocation
    base = arr
    offset = 0
    while base._ref is not None:
        # Calculate offset from parent
        offset += base.ptr - base._ref.ptr
        base = base._ref

    base_ptr = base.ptr
    base_size = base.capacity

    # Check if we already have this region
    if base_ptr in self.memory_regions:
        region = self.memory_regions[base_ptr]
        # Update role if needed (e.g., internal -> input)
        region.update_role(role)
    else:
        # Create new region
        region = MemoryRegion(
            region_id=len(self.memory_regions),
            size=base_size,
            device_ptr=base_ptr,
            role=role,
            initial_data=None,  # Filled during finalize
            base_array_ptr=base_ptr
        )
        self.memory_regions[base_ptr] = region

    # Map this array's ptr to the region
    self._ptr_to_region_id[arr.ptr] = region.region_id

    return region.region_id, offset
```

### 8.3 Reconstruction

On load, we:
1. Allocate memory for each region
2. Copy initial data
3. Reconstruct array views with correct offsets

```python
def _reconstruct_array_view(self, binding: ParamBinding) -> array_t:
    """Reconstruct an array_t for a kernel parameter."""

    region = self._memory_regions[binding.region_id]
    base_ptr = region.device_ptr

    return array_t(
        data=base_ptr + binding.offset,
        grad=0,  # No gradients in loaded graphs
        ndim=len(binding.shape),
        shape=binding.shape,
        strides=binding.strides
    )
```

---

## 9. Module and Kernel Binary Handling

### 9.1 Module CUBIN Files

In Warp, each Module contains multiple kernels. We store one .cubin file per Module (not per kernel) in standard CUDA binary format. This approach:
- Matches Warp's compilation model (one module = one cubin)
- Uses standard CUDA tooling (compatible with `cuobjdump`, etc.)
- Avoids redundant copies of shared code
- Allows sharing modules between multiple graphs

**File Organization:**
```
my_graph.wgf
my_graph_modules/
    simulation_abc12345.cubin    # Module "simulation" with hash abc12345
    rendering_def67890.cubin     # Module "rendering" with hash def67890
```

**Metadata in .wgf:**
```json
{
  "modules": [
    {
      "module_name": "simulation",
      "module_hash": "abc12345",
      "cubin_file": "simulation_abc12345.cubin",
      "target_arch": 86
    },
    {
      "module_name": "rendering",
      "module_hash": "def67890",
      "cubin_file": "rendering_def67890.cubin",
      "target_arch": 86
    }
  ],
  "kernels": [
    {
      "key": "simulation.integrate",
      "module_hash": "abc12345",
      "forward_name": "simulation_integrate_abc12345_cuda_kernel_forward",
      "backward_name": "simulation_integrate_abc12345_cuda_kernel_backward",
      "forward_smem_bytes": 0,
      "backward_smem_bytes": 0,
      "block_dim": 256
    },
    {
      "key": "simulation.collide",
      "module_hash": "abc12345",
      "forward_name": "simulation_collide_abc12345_cuda_kernel_forward",
      "backward_name": null,
      "forward_smem_bytes": 1024,
      "backward_smem_bytes": 0,
      "block_dim": 128
    },
    {
      "key": "rendering.shade",
      "module_hash": "def67890",
      "forward_name": "rendering_shade_def67890_cuda_kernel_forward",
      "backward_name": null,
      "forward_smem_bytes": 0,
      "backward_smem_bytes": 0,
      "block_dim": 256
    }
  ]
}
```

### 9.2 Architecture Compatibility

The .wgf file records the target SM architecture. On load:
1. If current device matches target arch: load CUBIN directly
2. If architecture differs: attempt to load (may fail if incompatible)

Future enhancement: store PTX alongside CUBIN for cross-architecture portability.

### 9.3 Loading Modules and Kernels

```python
def _load_modules(self, modules_dir: str):
    """Load CUBIN files for all modules."""

    # Load each unique module once
    for info in self._module_infos:
        cubin_path = os.path.join(modules_dir, info.cubin_file)

        # Load CUBIN as CUDA module
        module_handle = runtime.core.wp_cuda_load_module(
            self.device.context, cubin_path.encode()
        )

        if module_handle is None:
            raise RuntimeError(
                f"Failed to load module {info.module_name} from {cubin_path}"
            )

        self._loaded_modules[info.module_hash] = module_handle

def _load_kernels(self):
    """Get kernel function handles from loaded modules."""

    for info in self._kernel_infos:
        # Get the module handle for this kernel
        module_handle = self._loaded_modules[info.module_hash]

        # Get kernel function handles
        forward = runtime.core.wp_cuda_get_kernel(
            self.device.context, module_handle, info.forward_name.encode()
        )

        backward = None
        if info.backward_name:
            backward = runtime.core.wp_cuda_get_kernel(
                self.device.context, module_handle, info.backward_name.encode()
            )

        self._loaded_kernels[info.kernel_key] = LoadedKernel(
            forward=forward,
            backward=backward,
            smem_forward=info.forward_smem_bytes,
            smem_backward=info.backward_smem_bytes,
            block_dim=info.block_dim
        )
```

---

## 10. Error Handling

### 10.1 Capture-Time Errors

```python
class CaptureError(Exception):
    """Error during API capture."""
    pass

class UnsupportedOperationError(CaptureError):
    """Operation cannot be captured."""
    pass

# Example: Unsupported operation
def record_operation(self, op_type, **kwargs):
    if op_type not in SUPPORTED_OPS:
        raise UnsupportedOperationError(
            f"Operation '{op_type}' cannot be captured for serialization"
        )
```

### 10.2 Load-Time Errors

```python
class LoadError(Exception):
    """Error loading a .wgf file."""
    pass

class VersionMismatchError(LoadError):
    """File format version is not supported."""
    pass

class ArchitectureMismatchError(LoadError):
    """Target CUDA architecture is not compatible."""
    pass

class BindingError(Exception):
    """Error binding arrays to a loaded graph."""
    pass
```

---

## 11. Testing Strategy

### 11.1 Unit Tests

```python
# test_apic.py

def test_simple_kernel_capture():
    """Test capturing a single kernel launch."""

def test_array_aliasing():
    """Test that sliced arrays share memory regions."""

def test_save_load_roundtrip():
    """Test saving and loading produces equivalent results."""

def test_input_output_bindings():
    """Test binding different arrays to loaded graph."""

def test_ptx_fallback():
    """Test loading on different GPU architecture."""
```

### 11.2 Integration Tests

```python
def test_multi_kernel_pipeline():
    """Test capturing multiple dependent kernel launches."""

def test_large_memory():
    """Test with large arrays to verify memory handling."""

def test_cpp_header_compilation():
    """Test that generated C++ header compiles."""
```

---

## 12. Future Extensions

### 12.1 Conditional Graphs

Support for `wp.capture_if()` / `wp.capture_while()` conditional nodes.

### 12.2 Multi-GPU

Extend to support graphs spanning multiple devices.

### 12.3 Graph Optimization

Post-load optimization passes (kernel fusion, memory reuse).

### 12.4 Debugging Tools

- Graph visualization (DOT export)
- Profiling integration
- Memory usage analysis

---

## 13. Appendix

### 13.1 File Format Magic Numbers

```
WGF1 = 0x31464757 (little-endian "WGF1")
```

### 13.2 Data Type Handling

Rather than encoding specific Warp types, we use a **byte-size approach** that handles arbitrary vector/matrix types:

```c
// Array region stores only what's needed for memory operations
struct ArrayRegionInfo {
    uint32_t region_id;
    uint64_t size_bytes;      // Total allocation size
    uint32_t element_size;    // Bytes per element
    // ...
};
```

**Examples:**

| Warp Type | element_size |
|-----------|--------------|
| `float32` | 4 |
| `float64` | 8 |
| `vec3f` | 12 |
| `mat44f` | 64 |
| `vec(8, float16)` | 16 |
| `mat(3, 5, float32)` | 60 |

This approach:
- Works with any vector/matrix dimensions
- Works with custom struct types (just total byte size)
- Avoids complex type reflection/serialization
- Kernel code handles type interpretation

For binding validation, we only check `element_size` matches between the binding and the provided array.

### 13.3 CUDA Graph Node Updates

When binding arrays at runtime, we use CUDA's graph update APIs:

```cpp
// Update kernel node parameters without rebuilding graph
cudaGraphExecKernelNodeSetParams(
    graphExec,
    kernelNode,
    &updatedParams
);

// For memory operations
cudaGraphExecMemcpyNodeSetParams(graphExec, memcpyNode, &updatedParams);
```

This allows efficient rebinding without recreating the graph executable.

### 13.4 References

- CUDA Graph Management: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs
- CUDA Driver API: https://docs.nvidia.com/cuda/cuda-driver-api/
- cudaGraphExecKernelNodeSetParams: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html
- Warp Documentation: https://nvidia.github.io/warp/
