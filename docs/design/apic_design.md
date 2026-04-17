---
orphan: true
---

# APIC (API Capture) Design Document
## CUDA Graph Capture, Serialization, and Replay for Warp

**Version:** 1.6
**Date:** February 2026
**Status:** Implemented (Phases 1-4 Complete, Mesh Serialization)

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

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-1 | Capture kernel launches during `wp.capture_begin()` / `wp.capture_end()` | High | Done |
| FR-2 | Capture memory operations (memcpy, memset, allocations) | High | Done |
| FR-3 | Serialize captured graph to a custom binary format with `wp.capture_save()` | High | Done |
| FR-4 | Deserialize and recreate graph with `wp.capture_load()` | High | Done |
| FR-5 | Execute deserialized graph with `wp.capture_launch()` | High | Done |
| FR-6 | Support `wp.capture_func(fn)` convenience API for capturing a callable | Medium | Not Implemented |
| FR-7 | Serialize all referenced `wp.array` memory with proper aliasing handling | High | Done |
| FR-8 | Serialize compiled CUDA kernels (CUBIN as separate files) | High | Done |
| FR-9 | Generate C++ header for native application embedding | Medium | Partial (C++ loading API done) |
| FR-10 | Support input/output array designation for graph parameters | High | Done |
| FR-11 | Support `wp.Mesh`, `wp.Volume`, `wp.BVH` data structures | Medium | Mesh: Done |
| FR-12 | Handle array slicing/aliasing (same underlying memory) | High | Done |

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

### 3.5 Handle Type and Object Pointers

**Location:** `warp/_src/types.py`

Warp uses opaque 64-bit handles to reference native objects like `wp.Mesh`, `wp.Volume`, and `wp.BVH`. These handles are pointers to device-side structures that must be remapped when a serialized graph is loaded.

```python
class handle(uint64):
    """Type for object handles (Mesh, Volume, BVH) in kernel parameters.

    Behaves identically to uint64 but allows APIC to detect which params
    need pointer remapping during replay.
    """
    pass
```

**Usage in kernel signatures:**
```python
@wp.kernel
def query_mesh(mesh: wp.handle, points: wp.array(dtype=wp.vec3), ...):
    m = wp.mesh_get(mesh)  # Works same as uint64
    ...
```

**Handle locations in structs:**
```python
@wp.struct
class Body:
    mesh: wp.handle  # APIC auto-detects this field needs remapping
    transform: wp.mat44
```

The `wp.handle` type is distinct from `wp.uint64` at the type level (`param_type is wp.handle`), allowing APIC to automatically detect which kernel parameters and struct fields contain handles that need remapping.

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

#### 4.2.1 APIC Layer (Python-Only Implementation)

The APIC layer intercepts Warp operations during capture. **Implementation Note:** The actual implementation is purely Python-based, recording operations via hooks in `context.py` during kernel launches and memory operations. No C++ state management is required.

**Python APICapture Class (warp/_src/apic/capture.py):**

```python
class APICapture:
    """Records API calls during CUDA graph capture for serialization."""

    def __init__(self, device: Device, stream: Stream = None):
        self.device = device
        self.stream = stream or device.stream

        # Recorded data
        self.launches: list[LaunchRecord] = []
        self.memory_ops: list = []  # MemcpyRecord or MemsetRecord
        self.operations: list = []  # All ops in order: ("launch", idx) or ("memop", idx)
        self.memory_regions: dict[int, MemoryRegion] = {}  # base_ptr -> region
        self.modules: dict[str, ModuleInfo] = {}  # module_hash -> ModuleInfo
        self.kernels: dict[str, KernelInfo] = {}  # kernel_key -> KernelInfo

        # Input/output bindings
        self.input_bindings: dict[str, int] = {}  # name -> region_id
        self.output_bindings: dict[str, int] = {}  # name -> region_id

        # Internal tracking
        self._ptr_to_region_id: dict[int, int] = {}
        self._next_region_id: int = 0
        self._recording: bool = False

    def begin(self):
        """Start APIC recording."""
        self._recording = True

    def end(self):
        """End APIC recording."""
        self._recording = False

    def record_launch(self, launch, inputs=None, outputs=None):
        """Record a kernel launch from a Launch object."""
        # Records LaunchRecord with kernel info and param bindings
        # Tracks modules and kernels for serialization
        ...

    def record_memcpy_d2d(self, dest, dest_offset, src, src_offset, count):
        """Record a device-to-device memory copy operation."""
        # Creates MemcpyRecord and adds to operations list
        ...
```

**Operation Ordering:**

The implementation maintains operation order through an `operations` list that tracks both kernel launches and memory operations in their original sequence:

```python
# After recording a launch
self.operations.append(("launch", len(self.launches) - 1))

# After recording a memory op
self.operations.append(("memop", len(self.memory_ops) - 1))
```

This ensures correct execution order when operations are interleaved (e.g., kernel → memcpy → kernel).

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
| Metadata Section  |  (Binary: modules, kernels, bindings)
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
| METADATA | 0x01 | Binary metadata (modules, kernels, bindings) |
| MEMORY | 0x02 | Initial memory contents (or external file refs) |
| OPERATIONS | 0x03 | Serialized operation sequence |

**Metadata Section Format (Binary, Version 3):**

The metadata section uses a compact binary format with length-prefixed strings:

```c
// Metadata header
struct MetadataHeader {
    uint32_t version;           // Format version (3)
    uint32_t target_arch;       // CUDA SM version
    uint32_t num_modules;
    uint32_t num_kernels;
    uint32_t num_bindings;
};

// Module entry (repeated num_modules times)
struct ModuleEntry {
    uint32_t module_hash_len;   // String length
    char module_hash[];         // Module hash string
    uint32_t module_name_len;
    char module_name[];
    uint32_t cubin_filename_len;
    char cubin_filename[];
    uint32_t target_arch;
};

// Kernel entry (repeated num_kernels times)
struct KernelEntry {
    uint32_t kernel_key_len;
    char kernel_key[];
    uint32_t module_hash_len;
    char module_hash[];         // References ModuleEntry
    uint32_t forward_name_len;
    char forward_name[];
    uint32_t backward_name_len; // 0 if no backward kernel
    char backward_name[];
    uint32_t forward_smem_bytes;
    uint32_t backward_smem_bytes;
    uint32_t block_dim;
};

// Binding entry (repeated num_bindings times)
struct BindingEntry {
    uint32_t name_len;
    char name[];
    uint32_t region_id;
};
```

This binary format replaced the previous JSON-based metadata in version 3, providing:
- ~200 fewer lines of parsing code
- Type-safe C API for registration
- No JSON dependency in C++ code

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

Rather than creating a new `LoadedGraph` class, we extend the existing `Graph` class to support serialization and loading. This maintains API consistency and allows seamless use of loaded graphs with existing code (same `wp.capture_launch()` works for both).

**Actual Implementation:**

```python
class Graph:
    """Warp CUDA graph - extended to support serialization."""

    def __init__(self, device: Device, capture_id: int | None = None, apic_capture=None):
        self.device = device
        self.capture_id = capture_id
        self.module_execs: set[ModuleExec] = set()
        self.graph_exec: ctypes.c_void_p | None = None
        self.graph: ctypes.c_void_p | None = None
        # APIC capture state (for graphs being captured)
        self.apic_capture = apic_capture

        # APIC loaded state (populated when loaded from file - Python path)
        self._loaded_modules: dict = {}  # module_hash -> {cuda_module, info}
        self._memory_regions: dict = {}  # region_id -> {ptr, size, element_size, role}
        self._launches: list = []         # parsed launch records
        self._memory_ops: list = []       # parsed memory operations
        self._operations: list = []       # ordered ("launch", idx) or ("memop", idx)
        self._input_bindings: dict = {}   # name -> region_id (simple int mapping)
        self._output_bindings: dict = {}  # name -> region_id
        self._source_path: str | None = None
        self._kernel_cache: dict = {}     # (kernel_name, module_hash) -> kernel func
        self._metadata: dict = {}
        self._needs_rebuild: bool = False  # True if bindings changed

        # Native C++ graph handle (for C++ loading path)
        self._native_graph: ctypes.c_void_p | None = None

    @classmethod
    def load(cls, path: str, device: Device = None) -> "Graph":
        """Load a graph from a .wgf file (internal, use wp.capture_load() instead)."""
        from warp._src.apic.serialize import load_graph_into
        device = device or warp.get_device()
        graph = cls(device)
        graph._source_path = path
        load_graph_into(graph, path)
        return graph

    def bind_input(self, name: str, arr) -> None:
        """Bind an input array to a named input slot."""
        if name not in self._input_bindings:
            raise ValueError(f"Unknown input binding: {name}")
        region_id = self._input_bindings[name]
        region = self._memory_regions[region_id]
        if arr.capacity != region["size"]:
            raise ValueError(f"Size mismatch: expected {region['size']}, got {arr.capacity}")
        region["ptr"] = arr.ptr
        region["external"] = True
        self._update_region_bindings(region_id, arr.ptr)

    def bind_output(self, name: str, arr) -> None:
        """Bind an output array to a named output slot."""
        # Same logic as bind_input

    def _update_region_bindings(self, region_id: int, new_ptr: int):
        """Update all bindings referencing a region and mark for rebuild."""
        if self.is_loaded:
            self._needs_rebuild = True
            if self.graph_exec is not None:
                runtime.core.wp_cuda_graph_exec_destroy(...)
                self.graph_exec = None
        # Update kernel param bindings and memory ops with new pointer

    def _rebuild_cuda_graph(self):
        """Rebuild CUDA graph by replaying operations during capture."""
        # Destroy old graph
        # Start capture: wp_cuda_graph_begin_capture(..., external=0)
        # Replay: self._execute_loaded(stream)
        # End capture: wp_cuda_graph_end_capture()
        # self._needs_rebuild = False

    def _execute_loaded(self, stream=None):
        """Execute/replay operations (used during rebuild or execution)."""
        # For each op in self._operations:
        #   - memop: call wp_memcpy_* or wp_memset_device
        #   - launch: build args, call wp_cuda_launch_kernel

    @property
    def inputs(self) -> dict:
        """Get input binding names to region IDs."""
        return dict(self._input_bindings)

    @property
    def outputs(self) -> dict:
        """Get output binding names to region IDs."""
        return dict(self._output_bindings)

    @property
    def is_loaded(self) -> bool:
        """True if this graph was loaded from a file."""
        return self._source_path is not None
```

**Graph Reconstruction via Replay:**

When a graph is loaded or bindings change, we reconstruct the CUDA graph by replaying operations during capture:

```python
def _rebuild_cuda_graph(self):
    """Reconstruct CUDA graph from recorded operations."""
    # Destroy old graph/exec
    if self.graph is not None:
        runtime.core.wp_cuda_graph_destroy(self.device.context, self.graph)
    # ...

    stream = self.device.stream

    # Start capture (external=0 for new capture)
    runtime.core.wp_cuda_graph_begin_capture(self.device.context, stream.cuda_stream, 0)

    try:
        # Replay all operations - they get captured into the graph
        self._execute_loaded(stream)

        # End capture
        g = ctypes.c_void_p()
        runtime.core.wp_cuda_graph_end_capture(self.device.context, stream.cuda_stream, ctypes.byref(g))
        self.graph = g
        self._needs_rebuild = False

    except Exception:
        # Clean up capture state on failure
        runtime.core.wp_cuda_graph_end_capture(...)
        raise
```

#### 4.2.5 Input/Output Binding System

**Actual Implementation (Simplified):**

Rather than complex dataclasses with dtype/shape, the implementation uses simple mappings:

```python
# In APICapture during capture:
self.input_bindings: dict[str, int] = {}   # name -> region_id
self.output_bindings: dict[str, int] = {}  # name -> region_id

# Set during capture via:
apic_capture.set_input_binding(name, arr)   # Tracks region_id, marks role=INPUT
apic_capture.set_output_binding(name, arr)  # Tracks region_id, marks role=OUTPUT
```

**Binding Update Flow:**

When a user binds an array, we:
1. Validate array size matches region size
2. Update the region's pointer to point to the new array
3. Mark the array as externally owned (don't free on Graph destruction)
4. Update all parameter bindings referencing the region
5. Mark graph for rebuild (`_needs_rebuild = True`)

```python
def bind_input(self, name: str, arr) -> None:
    region_id = self._input_bindings[name]
    region = self._memory_regions[region_id]

    # Validate size (not dtype - we're type-agnostic via element_size)
    if arr.capacity != region["size"]:
        raise ValueError(f"Size mismatch")

    # Update region pointer
    region["ptr"] = arr.ptr
    region["external"] = True

    # Update all param bindings referencing this region
    self._update_region_bindings(region_id, arr.ptr)
```

**Note:** The current implementation validates only byte size, not dtype. This is intentional - it allows flexibility (e.g., binding a `float32[16]` where `vec4f[4]` was used) as long as total bytes match. Type interpretation is handled by the kernels.

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

**Implemented APIs:**

```python
# Public APIs exposed via warp module
def wp.capture_save(
    graph: Graph,
    path: str,
    inputs: dict[str, array] = None,
    outputs: dict[str, array] = None,
) -> None:
    """
    Save a captured CUDA graph to disk.

    Args:
        graph: A Graph as returned by wp.capture_end() with apic=True
        path: Output path (without extension). Creates {path}.wgf and {path}_modules/
        inputs: Dictionary mapping names to input arrays for binding
        outputs: Dictionary mapping names to output arrays for binding
    """

def wp.capture_load(path: str, device=None) -> Graph:
    """
    Load a previously saved CUDA graph from disk.

    Args:
        path: Path to the .wgf file (with or without extension)
        device: Target device (default: current CUDA device)

    Returns:
        Graph object ready for binding and execution via wp.capture_launch()
    """
```

**Usage Pattern:**

```python
# Capture with APIC enabled
with wp.ScopedCapture(apic=True) as capture:
    wp.launch(my_kernel, dim=n, inputs=[a], outputs=[b])

# Save with named bindings
wp.capture_save(capture.graph, "my_computation",
                inputs={"positions": a},
                outputs={"results": b})

# Load and execute
loaded_graph = wp.capture_load("my_computation")
loaded_graph.set_param("positions", new_positions)
loaded_graph.set_param("results", new_results)
wp.capture_launch(loaded_graph)
```

**Not Yet Implemented:**

```python
# capture_func() - convenience function (FR-6, deferred)
def capture_func(fn: Callable,
                 inputs: dict[str, array],
                 outputs: dict[str, array],
                 path: str = None,
                 device: Device = None) -> Graph | None:
    """Convenience function to capture and optionally save a computation."""
    pass  # Not implemented
```

### 5.2 Extended capture_begin/capture_end

APIC recording is enabled by default. The `apic` parameter can be set to `False` to disable it if needed:

```python
def capture_begin(device: Device = None,
                  stream: Stream = None,
                  force_module_load: bool = None,
                  external: bool = False,
                  apic: bool = True) -> None:
    """
    Begin CUDA graph capture.

    Args:
        device: Target device
        stream: Stream to capture on
        force_module_load: Force loading all modules before capture
        external: Whether capture was started externally
        apic: Enable API capture for serialization support (default: True)
    """

def capture_end(device: Device = None, stream: Stream = None) -> Graph:
    """
    End CUDA graph capture.

    Returns:
        Graph object with serialization support (save/bind methods).
    """
```

### 5.3 Extended Graph Class API

The existing `Graph` class gained new methods:

```python
class Graph:
    # Existing methods (unchanged)
    def __init__(self, device: Device, capture_id: int | None = None, apic_capture=None): ...
    def retain_module_exec(self, module_exec: ModuleExec): ...

    # Python loading path (use wp.capture_load() instead)
    @classmethod
    def load(cls, path: str, device: Device = None) -> "Graph":
        """Load a graph from a .wgf file using Python implementation."""

    # C++ native loading path (for testing/embedded use)
    @classmethod
    def load_native(cls, path: str, device: Device = None) -> "Graph":
        """Load a graph from a .wgf file using C++ implementation."""

    # New instance methods for loaded graphs
    def bind_input(self, name: str, arr) -> None:
        """Bind an array to a named input slot (marks graph for rebuild)."""

    def bind_output(self, name: str, arr) -> None:
        """Bind an array to a named output slot (marks graph for rebuild)."""

    @property
    def inputs(self) -> dict:
        """Get input binding names to region IDs (empty if not loaded)."""

    @property
    def outputs(self) -> dict:
        """Get output binding names to region IDs (empty if not loaded)."""

    @property
    def is_loaded(self) -> bool:
        """True if this graph was loaded from a file."""

    # Internal methods (Python path)
    def _rebuild_cuda_graph(self) -> None:
        """Rebuild CUDA graph by replaying operations during capture."""

    def _execute_loaded(self, stream=None) -> None:
        """Execute/replay loaded operations."""

    def _update_region_bindings(self, region_id: int, new_ptr: int) -> None:
        """Update all bindings referencing a region."""

    def _get_kernel_function(self, kernel_name: str, module_hash: str):
        """Get kernel function pointer from loaded modules."""

    # Properties
    @property
    def is_native(self) -> bool:
        """True if this graph was loaded via C++ native path."""
        return self._native_graph is not None
```

**Native Graph Handling:**

When a graph is loaded via C++, parameter updates use memcpy to the pre-allocated memory regions:

```python
def set_param(self, name: str, arr) -> None:
    if self._native_graph is not None:
        # Use C++ memcpy-based parameter setting
        result = runtime.core.wp_apic_set_param(
            self._native_graph, name.encode(), arr.ptr, arr.capacity
        )
        if result == 0:
            raise RuntimeError(f"Failed to set param: {runtime.get_error_string()}")
        return

def __del__(self):
    if self._native_graph is not None:
        runtime.core.wp_apic_destroy_graph(self._native_graph)
        self._native_graph = None
```

### 5.4 C++ Native API

The C++ APIC API enables both recording/saving and loading/executing serialized graphs without Python. All APIC declarations are consolidated in `apic.h`.

**C++ API (warp/native/apic.h):**

```cpp
// =============================================================================
// Recording API - Called during graph capture
// =============================================================================

// Opaque handle to APIC state (used during capture)
typedef struct APICStateInternal* APICState;

// State management
WP_API APICState wp_apic_create_state();
WP_API void wp_apic_destroy_state(APICState state);

// Recording control
WP_API void wp_apic_begin_recording(APICState state);
WP_API void wp_apic_end_recording(APICState state);

// Metadata registration - call these before wp_apic_state_save()
WP_API void wp_apic_register_module(
    APICState state,
    const char* module_hash,
    const char* module_name,
    const char* cubin_filename,
    int target_arch);

WP_API void wp_apic_register_kernel(
    APICState state,
    const char* kernel_key,
    const char* module_hash,
    const char* forward_name,
    const char* backward_name,  // can be NULL
    int forward_smem_bytes,
    int backward_smem_bytes,
    int block_dim);

WP_API void wp_apic_register_binding(
    APICState state,
    const char* name,
    uint32_t region_id);

// Save state to .wgf file (serializes registered metadata to binary format)
WP_API int wp_apic_state_save(APICState state, const char* path, uint32_t target_arch);

// =============================================================================
// Loading API - Load and execute serialized graphs
// =============================================================================

// Opaque handle for a loaded APIC graph
typedef struct APICGraphInternal* APICGraph;

// Load a graph from a .wgf file
// Returns: Graph handle on success, nullptr on failure
WP_API APICGraph wp_apic_load_graph(void* context, const char* path);

// Destroy a loaded graph and free all associated resources
WP_API void wp_apic_destroy_graph(APICGraph graph);

// Set parameter data by copying to the pre-allocated memory region
// No graph rebuild is needed - data is copied directly via memcpy
// Returns: 1 on success, 0 on failure
WP_API int wp_apic_set_param(APICGraph graph, const char* name,
                              const void* data, size_t size);

// Get the device pointer for a parameter's pre-allocated region
// Returns: Device pointer or nullptr if not found
WP_API void* wp_apic_get_param_ptr(APICGraph graph, const char* name);

// Get the CUDA graph (builds on first access)
WP_API void* wp_apic_get_cuda_graph(APICGraph graph);

// Get the instantiated CUDA graph exec
WP_API void* wp_apic_get_cuda_graph_exec(APICGraph graph);

// Query functions
WP_API int wp_apic_get_num_params(APICGraph graph);
WP_API const char* wp_apic_get_param_name(APICGraph graph, int index);
WP_API size_t wp_apic_get_param_size(APICGraph graph, const char* name);
```

**Python ctypes Bindings (warp/_src/context.py):**

```python
# APIC graph loading functions
self.core.wp_apic_load_graph.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
self.core.wp_apic_load_graph.restype = ctypes.c_void_p

self.core.wp_apic_destroy_graph.argtypes = [ctypes.c_void_p]
self.core.wp_apic_destroy_graph.restype = None

self.core.wp_apic_set_param.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                         ctypes.c_void_p, ctypes.c_size_t]
self.core.wp_apic_set_param.restype = ctypes.c_int

self.core.wp_apic_get_param_ptr.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
self.core.wp_apic_get_param_ptr.restype = ctypes.c_void_p
```

**Graph.load_native() Method:**

For testing and hybrid workflows, the C++ loading path is accessible from Python:

```python
@classmethod
def load_native(cls, path: str, device: Device = None) -> "Graph":
    """Load a graph using the native C++ implementation.

    This uses the C++ APIC loading code path, which is useful for:
    - Testing the native loading implementation
    - Hybrid workflows where native loading is preferred

    Args:
        path: Path to the .wgf file
        device: Target CUDA device (default: current device)

    Returns:
        Graph object with native graph handle
    """
    device = device or warp.get_device()
    graph = cls(device)
    graph._native_graph = runtime.core.wp_apic_load_graph(
        device.context, path.encode()
    )
    if graph._native_graph is None:
        raise RuntimeError(f"Failed to load graph: {runtime.get_error_string()}")
    graph._source_path = path
    return graph
```

**C++ Implementation Details (warp/native/apic.cu):**

The C++ implementation uses an `APICGraphInternal` structure:

```cpp
struct APICGraphInternal {
    void* context;                        // CUDA context
    void* stream;                         // CUDA stream for execution
    void* graph;                          // cudaGraph_t
    void* graph_exec;                     // cudaGraphExec_t

    // Module management
    std::map<std::string, void*> modules; // module_hash -> CUmodule

    // Memory regions (pre-allocated, fixed addresses)
    std::map<int, MemoryRegion> regions;  // region_id -> {ptr, size}

    // Kernel cache
    std::map<std::pair<std::string, std::string>, void*> kernel_cache;

    // Binding maps (name -> region_id)
    std::map<std::string, int> bindings;

    // Operations for replay
    std::vector<LaunchRecord> launches;
    std::vector<MemoryOp> memory_ops;
    std::vector<std::pair<int, int>> operations;  // (op_type, idx)

    // Parsed metadata (binary format, no JSON)
    std::vector<APICModuleInfo> module_infos;
    std::vector<APICKernelInfo> kernel_infos;
};
```

Graph reconstruction uses the same capture-replay pattern as Python:

```cpp
int wp_apic_launch(wp_apic_graph_t graph_handle, void* stream) {
    LoadedApicGraph* graph = (LoadedApicGraph*)graph_handle;

    // Rebuild CUDA graph if bindings changed
    if (graph->needs_rebuild || graph->graph_exec == nullptr) {
        rebuild_cuda_graph(graph, stream);
    }

    // Launch the graph
    cudaGraphLaunch((cudaGraphExec_t)graph->graph_exec, (cudaStream_t)stream);
    return 1;
}
```

### 5.5 capture_launch (Unchanged)

The existing `capture_launch` works with all Graph objects:

```python
def capture_launch(graph: Graph, stream: Stream = None) -> None:
    """Launch a captured CUDA graph."""
    # Existing implementation - works for both regular and loaded graphs
```

---

## 6. Implementation Strategy

### 6.1 Phase 1: Python APIC Infrastructure

**Implementation Note:** The original design proposed C++ infrastructure, but the actual implementation is Python-only. This simplifies the implementation significantly while achieving the same functionality.

1. **APICapture class** - Python-side recording state and orchestration
2. **record_launch() hook** - Called from launch() during APIC-enabled capture
3. **Memory operation hooks** - Record memcpy/memset via Python callbacks
4. **begin()/end() methods** - Start/stop recording on APICapture object
5. **Memory region tracking** - Handle array aliasing via `_ref` chain resolution

### 6.2 Phase 2: Python APIC Layer (Arrays Only)

1. **Extend Graph class** - Add APIC fields for loaded state (`_loaded_modules`, `_memory_regions`, `_launches`, `_memory_ops`, `_operations`, `_input_bindings`, `_output_bindings`, `_source_path`, `_kernel_cache`, `_metadata`, `_needs_rebuild`)
2. **APICapture class** - Python recording state (warp/_src/apic/capture.py)
3. **Memory region tracking** - Handle array aliasing via `_ref` chain resolution in track_array()
4. **Launch recording** - Capture LaunchRecord objects with kernel info and parameter bindings
5. **WGF file format** - Binary format with header, sections (metadata JSON, memory, operations)
6. **save_graph()** - Serialize to .wgf + copy .cubin files (warp/_src/apic/serialize.py)
7. **Graph.load()** - Class method to load .wgf and reconstruct graph via capture replay

### 6.3 Phase 3: Input/Output Bindings

1. **Binding specification** - Mark arrays as inputs/outputs via `set_input_binding()`/`set_output_binding()` during capture
2. **Region-based tracking** - Bindings map names to region IDs (simple dict[str, int])
3. **Memcpy-based parameter updates** - When parameters change, data is copied directly to the pre-allocated memory region via `cudaMemcpy`. No graph rebuild is needed since the graph uses the same fixed memory addresses.
4. **Validation** - Size checking for bound arrays (capacity must match region size)
5. **Graph.set_param()** - Copy data to the pre-allocated region

**Design note:** The simplified approach uses fixed memory allocations and memcpy for parameter updates. This avoids the complexity of graph rebuilding and is efficient for frequent parameter changes.

### 6.4 Phase 4: C++ Native Loading (Implemented)

**Completed:**

1. **C++ APIC API** - Native functions in warp.h for loading, binding, and launching
2. **LoadedApicGraph structure** - C++ equivalent of Python's Graph class with loaded state
3. **WGF file parsing** - Header, section table, metadata (JSON), memory, and operations parsing
4. **Module loading** - CUBIN files loaded via wp_cuda_load_module
5. **Memory region management** - Allocation and binding with external array support
6. **Kernel lookup** - Function handles retrieved via cuModuleGetFunction_f wrapper
7. **Graph reconstruction** - Capture-replay pattern using CUDA graph APIs
8. **Python ctypes bindings** - Full integration with context.py
9. **Graph.load_native()** - Python method to test C++ loading path
10. **Test coverage** - test_apic_native_loading verifies C++ implementation

**Not Yet Implemented:**

1. **C++ header generation** - Generate standalone .h/.cpp for embedding without Warp runtime
2. **CMake integration** - Build system support for generated C++ code

### 6.5 Phase 5: Complex Data Structures

1. **wp.Mesh** - Serialize points, indices arrays (uses Phase 2 array support)
2. **wp.BVH** - Serialize bounds arrays
3. **wp.Volume** - Serialize NanoVDB data as uint8 array
4. **Struct arrays** - Already handled via element_size approach

---

## 7. Integration Points

### 7.1 Modifications to Existing Code

**Implementation Note:** The original design proposed C++ modifications. The actual implementation is Python-only, which simplifies integration significantly.

**warp/_src/context.py (Python Layer):**
- Extended `Graph` class with APIC fields:
  - `_loaded_modules`, `_memory_regions`, `_launches`, `_memory_ops`, `_operations`
  - `_input_bindings`, `_output_bindings` (dict[str, int] mapping names to region IDs)
  - `_source_path`, `_kernel_cache`, `_metadata`, `_needs_rebuild`
- Added `apic` parameter to `capture_begin()`
- Added `apic_capture` field to `Graph.__init__()`
- Modified `launch()` to call `apic_capture.record_launch()` when APIC active
- Added `Graph.load()` class method (internal, called by `wp.capture_load()`)
- Added `Graph.set_param()`, `Graph.get_param_ptr()` methods for parameter access
- Added `Graph.is_loaded` property
- Added public APIs: `wp.capture_save()`, `wp.capture_load()`

**warp/_src/types.py:**
- No modifications needed; `_ref` chain and array attributes work as-is

**warp/native/warp.h (C++ API):**
- Added `wp_apic_load_graph()` - Load graph from .wgf file
- Added `wp_apic_destroy_graph()` - Free graph resources
- Added `wp_apic_set_param()` - Set parameter data via memcpy (no graph rebuild needed)
- Added `wp_apic_get_param_ptr()` - Get device pointer for a parameter
- Added `wp_apic_get_cuda_graph()` - Get CUDA graph (builds on first access)
- Added `wp_apic_get_cuda_graph_exec()` - Get instantiated CUDA graph exec

**warp/native/warp.cu (C++ Implementation):**
- Added `LoadedApicGraph` structure with module map, memory regions, kernel cache, bindings, operations
- Added `apic_launch_bounds_t` and `apic_array_t` structures for kernel parameters
- Added WGF file parsing (header, sections, metadata JSON, memory, operations)
- Added module loading via `wp_cuda_load_module()`
- Added kernel lookup via `cuModuleGetFunction_f()` wrapper
- Added graph reconstruction via capture-replay pattern
- Uses CUDA runtime API (cudaMalloc, cudaMemcpy) to avoid linker issues with driver API

### 7.2 New Files (Actual Implementation)

```
warp/_src/apic/
    __init__.py           # Public exports (APICapture, KernelInfo, etc.)
    capture.py            # APICapture class, LaunchRecord, MemcpyRecord, etc.
    serialize.py          # save_graph(), load_graph(), load_graph_into()
    format.py             # WGFReader, WGFWriter for .wgf file handling
```

**C++ Native API (Phase 4):**

```
warp/native/apic.h           # All APIC API declarations (public C API + internal C++)
warp/native/apic_types.h     # POD structs for binary serialization
warp/native/apic.cu          # APIC implementation (recording, serialization, loading)
warp/native/warp.h           # Includes apic.h for backward compatibility
warp/native/warp.cu          # APICStateInternal struct, recording hooks
warp/_src/context.py         # ctypes bindings, Graph.load_native()
warp/tests/cuda/test_apic.py # Comprehensive test suite (18 tests)
```

**Not implemented (deferred):**
- `cpp_gen.py` - C++ header/implementation generation for standalone embedding

### 7.3 Python Recording Hooks

Recording is done at the Python level via explicit calls:

```python
# In context.py launch() function
def launch(...):
    # ... existing launch code ...

    # APIC recording hook
    if graph and graph.apic_capture and graph.apic_capture.is_recording():
        graph.apic_capture.record_launch(launch_obj, inputs, outputs)

    # ... rest of launch code ...
```

Memory operations are recorded via dedicated methods on APICapture:

```python
# Recording D2D copy
apic_capture.record_memcpy_d2d(dest, dest_offset, src, src_offset, count)
```

### 7.4 Graph Reconstruction

Rather than recording operations in C++ and retrieving them, the implementation:
1. Records operations at the Python level during capture
2. On load, parses the .wgf file and populates Graph fields
3. Rebuilds the CUDA graph by starting capture, replaying all operations, and ending capture

```python
def _rebuild_cuda_graph(self):
    """Rebuild CUDA graph by replaying operations during capture."""
    # Start capture
    runtime.core.wp_cuda_graph_begin_capture(self.device.context, stream.cuda_stream, 0)
    try:
        # Replay all recorded operations
        self._execute_loaded(stream)
        # End capture to get the graph
        runtime.core.wp_cuda_graph_end_capture(self.device.context, stream.cuda_stream, ctypes.byref(g))
        self.graph = g
    except:
        # Clean up on failure
        runtime.core.wp_cuda_graph_end_capture(...)
        raise
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

## 9. Handle Pointer Serialization

### 9.1 Problem Statement

Objects like `wp.Mesh`, `wp.Volume`, and `wp.BVH` are referenced by 64-bit handle pointers. When a graph is serialized and later loaded:
1. The original objects no longer exist
2. New objects must be created from serialized data
3. All handle pointers in kernel arguments and memory regions must be updated to point to the new objects

### 9.2 Handle Detection

APIC automatically detects handle locations through the `wp.handle` type:

**Kernel arguments:**
```python
@wp.kernel
def query_mesh(mesh: wp.handle, ...):  # Direct handle argument
    ...
```

**Struct fields:**
```python
@wp.struct
class Body:
    mesh: wp.handle  # Handle inside struct
    transform: wp.mat44
```

**Arrays of structs:**
```python
bodies = wp.zeros(n, dtype=Body)  # Array with handle fields
```

The `_find_handle_offsets()` function in `capture.py` recursively inspects types to find all byte offsets where handles are located:

```python
def _find_handle_offsets(self, dtype, base_offset=0) -> list[int]:
    """Recursively find byte offsets of wp.handle fields in a type."""
    offsets = []
    if dtype is wp.handle:
        offsets.append(base_offset)
    elif isinstance(dtype, wp.struct):
        for field_name, var in dtype.vars.items():
            field_offset = getattr(dtype.ctype, field_name).offset
            offsets.extend(self._find_handle_offsets(var.type, base_offset + field_offset))
    return offsets
```

### 9.3 Pointer Location Registration

When an array is tracked during capture, APIC automatically registers handle pointer locations:

```python
def track_array(self, arr) -> tuple[int, int]:
    region_id, offset = ...  # Register memory region

    # Auto-detect handle locations in the array's dtype
    handle_offsets = self._find_handle_offsets(arr.dtype)
    if handle_offsets:
        stride = wp.types.type_size_in_bytes(arr.dtype)
        for handle_offset in handle_offsets:
            runtime.core.wp_apic_register_ptr_location(
                self.native_state, region_id, handle_offset, stride
            )

    return region_id, offset
```

**C++ API for pointer location registration:**
```cpp
// Register a handle pointer location within a memory region
// offset: byte offset of first handle in the region
// stride: bytes between consecutive handles (0 for single pointer)
WP_API void wp_apic_register_ptr_location(
    APICState state,
    uint32_t region_id,
    uint64_t offset,
    uint64_t stride);
```

### 9.4 Mesh Serialization

Meshes are automatically discovered from the native `g_mesh_descriptors` registry during serialization:

```cpp
// In apic_serialize_metadata()
for (auto& [ptr, mesh] : wp::g_mesh_descriptors) {
    APICMeshRecord rec;
    rec.num_points = mesh.num_points;
    rec.num_tris = mesh.num_tris;
    rec.points_region_id = find_or_register_region(mesh.points.data, ...);
    rec.indices_region_id = find_or_register_region(mesh.indices.data, ...);
    rec.original_ptr = ptr;
    // ... write record
}
```

**APICMeshRecord structure:**
```cpp
#pragma pack(push, 1)
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
} APICMeshRecord;
#pragma pack(pop)
```

### 9.5 Handle Remapping During Load

When a graph is loaded:

1. **Memory regions are allocated** and data is copied from the serialized file
2. **Meshes are recreated** using `wp_mesh_create_device()` with the restored array data
3. **Handle remap table is built** mapping old pointers to new pointers
4. **Pointer fixup** updates all registered handle locations

**Building the remap table:**
```cpp
// Create mesh from serialized data
uint64_t new_mesh_id = wp_mesh_create_device(ctx, points, velocities, indices, ...);

// Add to remap table
graph->handle_ptr_remap[rec.original_ptr] = new_mesh_id;
```

**Fixing up kernel arguments (scalar handles):**
```cpp
// For uint64-sized scalar parameters, check for remapping
if (scalar_size == sizeof(uint64_t) && !graph->handle_ptr_remap.empty()) {
    uint64_t* handle_ptr = reinterpret_cast<uint64_t*>(scalar.get());
    auto remap_it = graph->handle_ptr_remap.find(*handle_ptr);
    if (remap_it != graph->handle_ptr_remap.end()) {
        *handle_ptr = remap_it->second;
    }
}
```

**Fixing up memory regions (handle arrays):**
```cpp
// Note: regions are in device memory, so must use cudaMemcpy
static void apic_fixup_ptr_locations(APICGraphInternal* graph) {
    for (const auto& loc : graph->ptr_locations) {
        uint8_t* base = (uint8_t*)graph->regions[loc.region_id].ptr;
        uint64_t region_size = graph->regions[loc.region_id].size;

        for (uint64_t off = loc.offset; off + sizeof(uint64_t) <= region_size;
             off += loc.stride) {
            uint8_t* device_ptr = base + off;
            uint64_t old_val;
            cudaMemcpy(&old_val, device_ptr, sizeof(uint64_t), cudaMemcpyDeviceToHost);

            auto remap_it = graph->handle_ptr_remap.find(old_val);
            if (remap_it != graph->handle_ptr_remap.end()) {
                uint64_t new_val = remap_it->second;
                cudaMemcpy(device_ptr, &new_val, sizeof(uint64_t), cudaMemcpyHostToDevice);
            }

            if (loc.stride == 0) break;  // Single pointer
        }
    }
}
```

### 9.6 Usage Examples

**Direct handle as kernel argument:**
```python
@wp.kernel
def query_mesh(mesh: wp.handle, query_points: wp.array(dtype=wp.vec3), ...):
    m = wp.mesh_get(mesh)
    ...

mesh = wp.Mesh(points, indices)

with wp.ScopedCapture(apic=True) as cap:
    wp.launch(query_mesh, inputs=[mesh.id, query_points, ...])

wp.capture_save(cap.graph, "mesh_query", inputs={"query_points": query_points}, ...)

# Later:
loaded = wp.capture_load("mesh_query")
wp.capture_launch(loaded)  # Mesh automatically recreated and remapped
```

**Handle in struct array:**
```python
@wp.struct
class Body:
    mesh: wp.handle
    transform: wp.mat44

bodies = wp.zeros(n, dtype=Body)
for i, m in enumerate(meshes):
    bodies[i].mesh = m.id

with wp.ScopedCapture(apic=True) as cap:
    wp.launch(kernel, inputs=[bodies, ...])

# Handle locations auto-detected from Body.mesh field
wp.capture_save(cap.graph, "simulation", inputs={"bodies": bodies}, ...)
```

---

## 10. Module and Kernel Binary Handling

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

**Metadata Registration (Python → C++):**

Python registers metadata via C API calls before saving:

```python
# Register modules
for module_hash, info in capture.modules.items():
    runtime.core.wp_apic_register_module(
        capture.native_state,
        module_hash.encode('utf-8'),
        info.module_name.encode('utf-8'),
        info.cubin_filename.encode('utf-8'),
        info.target_arch)

# Register kernels
for kernel_key, info in capture.kernels.items():
    runtime.core.wp_apic_register_kernel(
        capture.native_state,
        kernel_key.encode('utf-8'),
        info.module_hash.encode('utf-8'),
        info.forward_name.encode('utf-8'),
        (info.backward_name or "").encode('utf-8'),
        info.forward_smem_bytes,
        info.backward_smem_bytes,
        info.block_dim)

# Register bindings
for name, region_id in capture.bindings.items():
    runtime.core.wp_apic_register_binding(
        capture.native_state,
        name.encode('utf-8'),
        region_id)

# Save (C++ serializes to binary format internally)
runtime.core.wp_apic_state_save(capture.native_state, path, target_arch)
```

The C++ side stores registered data and serializes it to binary format. This replaces the previous JSON-based approach, eliminating ~300 lines of JSON parsing code.

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

## 11. Error Handling

### 11.1 Capture-Time Errors

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

### 11.2 Load-Time Errors

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

## 12. Testing Strategy

### 12.1 Unit Tests

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

### 12.2 Integration Tests

```python
def test_multi_kernel_pipeline():
    """Test capturing multiple dependent kernel launches."""

def test_large_memory():
    """Test with large arrays to verify memory handling."""

def test_cpp_header_compilation():
    """Test that generated C++ header compiles."""
```

### 12.3 C++ Native Loading Tests

```python
def test_apic_native_loading():
    """Test loading a graph using the native C++ implementation.

    Verifies:
    - wp.capture_load() successfully loads a .wgf file via C++ code path
    - set_param() copies data to pre-allocated memory via memcpy
    - wp.capture_launch() executes the graph and produces correct results
    - Graph cleanup via wp_apic_destroy_graph() works without errors
    """
```

**Test Coverage (19 tests total):**
- Basic capture/save/load roundtrips
- Array aliasing and slicing
- Input/output bindings via set_param()
- Multiple kernels and modules
- Memory operations (memcpy, memset, allocations)
- C++ native loading path

---

## 13. Future Extensions

### 13.1 Conditional Graphs

Support for `wp.capture_if()` / `wp.capture_while()` conditional nodes.

### 13.2 Multi-GPU

Extend to support graphs spanning multiple devices.

### 13.3 Graph Optimization

Post-load optimization passes (kernel fusion, memory reuse).

### 13.4 Debugging Tools

- Graph visualization (DOT export)
- Profiling integration
- Memory usage analysis

---

## 14. Appendix

### 14.1 File Format Magic Numbers

```
WGF1 = 0x31464757 (little-endian "WGF1")
```

### 14.2 Data Type Handling

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

### 14.3 CUDA Graph Node Updates

**Original Design (Not Implemented):**

The design proposed using CUDA's graph update APIs for efficient in-place binding updates:

```cpp
// Update kernel node parameters without rebuilding graph
cudaGraphExecKernelNodeSetParams(graphExec, kernelNode, &updatedParams);

// For memory operations
cudaGraphExecMemcpyNodeSetParams(graphExec, memcpyNode, &updatedParams);
```

**Actual Implementation:**

The current implementation takes a simpler approach: when bindings change, the entire CUDA graph is rebuilt by replaying all operations during capture:

```python
def bind_input(self, name: str, arr):
    # ... update region pointer ...
    self._needs_rebuild = True

# Later, in capture_launch() or explicitly:
if graph._needs_rebuild:
    graph._rebuild_cuda_graph()  # Replay all ops during capture
```

**Trade-offs:**

| Aspect | Node Update APIs | Graph Rebuild |
|--------|------------------|---------------|
| Performance | O(1) per binding change | O(n) where n = num operations |
| Complexity | Requires tracking graph nodes | Simple replay mechanism |
| CUDA API | Uses advanced graph APIs | Uses basic capture APIs |
| Flexibility | Limited to parameter changes | Can handle structural changes |

For graphs with many operations and frequent binding changes, the node update approach would be more efficient. For typical use cases (bind once, launch many times), the rebuild approach is sufficient and simpler to implement.

### 14.4 References

- CUDA Graph Management: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs
- CUDA Driver API: https://docs.nvidia.com/cuda/cuda-driver-api/
- cudaGraphExecKernelNodeSetParams: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html
- Warp Documentation: https://nvidia.github.io/warp/
