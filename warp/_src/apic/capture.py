# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
APIC Capture - Records API calls during CUDA graph capture for serialization.

Recording is done natively in C++ (warp.cu/apic.cu). Python's role is to:
- Track arrays and compute region_id/offset
- Track modules and kernels for metadata
- Compute parameter bindings from Python type info
- Set launch context before each kernel launch
"""

import ctypes
from dataclasses import dataclass

# Constants from apic_types.h
APIC_MAX_DIMS = 4
APIC_LAUNCH_MAX_DIMS = 4
APIC_MAX_SCALAR_SIZE = 128
APIC_PARAM_ARRAY = 1
APIC_PARAM_SCALAR = 2


# Memory region roles (must match C++ APICMemoryRole enum)
class MemoryRole:
    INTERNAL = 0
    INPUT = 1
    OUTPUT = 2
    INPUT_OUTPUT = 3


class APICParamBindingInfo(ctypes.Structure):
    """Parameter binding info - matches C struct in apic_types.h."""

    _pack_ = 1
    _fields_ = [  # noqa: RUF012
        ("type", ctypes.c_uint8),
        ("ndim", ctypes.c_uint8),
        ("param_index", ctypes.c_uint16),
        ("region_id", ctypes.c_int32),
        ("byte_offset", ctypes.c_uint64),
        ("shape", ctypes.c_int64 * APIC_MAX_DIMS),
        ("strides", ctypes.c_int64 * APIC_MAX_DIMS),
        ("element_size", ctypes.c_uint32),
        ("scalar_size", ctypes.c_uint16),
        ("_pad", ctypes.c_uint8 * 2),
        ("scalar_value", ctypes.c_uint8 * APIC_MAX_SCALAR_SIZE),
    ]


class APICLaunchInfo(ctypes.Structure):
    """Launch info passed to wp_cuda_launch_kernel() - matches C struct in apic_types.h."""

    _pack_ = 1
    _fields_ = [  # noqa: RUF012
        ("kernel_key", ctypes.c_char_p),
        ("module_hash", ctypes.c_char_p),
        ("shape", ctypes.c_int32 * APIC_LAUNCH_MAX_DIMS),
        ("ndim", ctypes.c_int32),
        ("block_dim", ctypes.c_int32),
        ("smem_bytes", ctypes.c_int32),
        ("is_forward", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8 * 3),
        ("params", ctypes.POINTER(APICParamBindingInfo)),
        ("num_params", ctypes.c_int32),
    ]


@dataclass
class MemoryRegion:
    """Represents a contiguous memory allocation."""

    region_id: int
    size: int  # Size in bytes
    device_ptr: int  # Original device pointer
    element_size: int  # Size of one element in bytes
    role: int = MemoryRole.INTERNAL

    def update_role(self, new_role: int):
        """Update role, preferring higher priority roles."""
        if new_role > self.role:
            self.role = new_role
        elif self.role == MemoryRole.INPUT and new_role == MemoryRole.OUTPUT:
            self.role = MemoryRole.INPUT_OUTPUT
        elif self.role == MemoryRole.OUTPUT and new_role == MemoryRole.INPUT:
            self.role = MemoryRole.INPUT_OUTPUT


@dataclass
class ModuleInfo:
    """Module metadata - one cubin file contains all kernels from a module."""

    module_name: str
    module_hash: str
    target_arch: int
    cubin_filename: str = ""


@dataclass
class KernelInfo:
    """Kernel metadata - references its containing module."""

    kernel_key: str
    module_hash: str
    forward_name: str
    backward_name: str | None
    forward_smem_bytes: int
    backward_smem_bytes: int
    block_dim: int


class APICapture:
    """Records API calls during CUDA graph capture for serialization.

    Recording happens natively in C++. Python's role is to:
    - Track arrays and compute region_id/offset
    - Track modules and kernels for metadata
    - Build APICLaunchInfo structs with parameter bindings
    - Pass launch info directly to wp_cuda_launch_kernel()
    """

    def __init__(self, device, stream=None):
        self.device = device
        self.stream = stream or device.stream

        # Native APIC state handle
        self.native_state = None

        # Metadata (still tracked in Python for building metadata JSON)
        self.memory_regions: dict[int, MemoryRegion] = {}  # base_ptr -> region
        self.modules: dict[str, ModuleInfo] = {}  # module_hash -> ModuleInfo
        self.kernels: dict[str, KernelInfo] = {}  # kernel_key -> KernelInfo

        # Input/output bindings (set during save)
        self.input_bindings: dict[str, int] = {}  # name -> region_id
        self.output_bindings: dict[str, int] = {}  # name -> region_id

        # Internal tracking
        self._ptr_to_region_id: dict[int, int] = {}  # any ptr -> region_id
        self._next_region_id: int = 0
        self._recording: bool = False

    def begin(self):
        """Start APIC recording."""
        import warp._src.context

        runtime = warp._src.context.runtime

        # Create native state
        self.native_state = runtime.core.wp_apic_create_state()
        if not self.native_state:
            raise RuntimeError("Failed to create APIC state")

        # Begin native recording
        runtime.core.wp_apic_begin_recording(self.native_state)
        self._recording = True

    def end(self):
        """End APIC recording."""
        import warp._src.context

        if self.native_state:
            runtime = warp._src.context.runtime
            runtime.core.wp_apic_end_recording(self.native_state)
        self._recording = False

    def destroy(self):
        """Destroy native state and free resources."""
        import warp._src.context

        if self.native_state:
            runtime = warp._src.context.runtime
            runtime.core.wp_apic_destroy_state(self.native_state)
            self.native_state = None

    def is_recording(self) -> bool:
        """Check if recording is active."""
        return self._recording

    @property
    def operation_count(self) -> int:
        """Get the number of recorded operations (launches, memops, allocs)."""
        import warp._src.context

        if self.native_state:
            runtime = warp._src.context.runtime
            return runtime.core.wp_apic_get_operation_count(self.native_state)
        return 0

    @property
    def module_count(self) -> int:
        """Get the number of unique modules recorded."""
        return len(self.modules)

    @property
    def kernel_count(self) -> int:
        """Get the number of unique kernels recorded."""
        return len(self.kernels)

    def track_array(self, arr, role: int = MemoryRole.INTERNAL) -> tuple[int, int]:
        """
        Track an array, resolving to its base allocation.
        Registers the region with native code if new.

        Returns:
            (region_id, offset) - The region ID and byte offset within the region
        """
        import warp._src.context

        # Walk the _ref chain to find base allocation
        base = arr
        offset = 0
        while hasattr(base, "_ref") and base._ref is not None:
            offset += base.ptr - base._ref.ptr
            base = base._ref

        base_ptr = base.ptr
        base_size = base.capacity
        element_size = self._get_element_size(arr)

        # Check if we already have this region
        if base_ptr in self.memory_regions:
            region = self.memory_regions[base_ptr]
            region.update_role(role)
            return region.region_id, offset

        # Register with native code
        runtime = warp._src.context.runtime
        region_id = runtime.core.wp_apic_register_memory_region(
            self.native_state,
            base_ptr,
            base_size,
            element_size,
            role,
        )

        # Track in Python for metadata
        region = MemoryRegion(
            region_id=region_id,
            size=base_size,
            device_ptr=base_ptr,
            element_size=element_size,
            role=role,
        )
        self.memory_regions[base_ptr] = region
        self._ptr_to_region_id[arr.ptr] = region_id
        self._next_region_id = max(self._next_region_id, region_id + 1)

        return region_id, offset

    def _get_element_size(self, arr) -> int:
        """Get the size of one element in bytes."""
        dtype = arr.dtype
        if hasattr(dtype, "_type_size_"):
            return dtype._type_size_
        elif hasattr(dtype, "_length_") and hasattr(dtype, "_type_"):
            return dtype._length_ * ctypes.sizeof(dtype._type_)
        else:
            return ctypes.sizeof(dtype)

    def build_launch_info(self, launch, inputs=None, outputs=None) -> APICLaunchInfo:
        """Build APICLaunchInfo struct for a kernel launch.

        Tracks metadata and memory regions, then builds the launch info struct
        to be passed to wp_cuda_launch_kernel().

        Args:
            launch: The Launch object containing kernel and parameter info
            inputs: Original input arrays (for memory region tracking)
            outputs: Original output arrays (for memory region tracking)

        Returns:
            APICLaunchInfo struct to pass to wp_cuda_launch_kernel()
        """
        kernel = launch.kernel
        module = kernel.module
        module_exec = launch.module_exec
        module_hash = (
            module_exec.module_hash.hex() if isinstance(module_exec.module_hash, bytes) else module_exec.module_hash
        )
        module_hash_short = module_hash[:8]

        # Track memory regions from original arrays
        all_arrays = []
        if inputs:
            all_arrays.extend(inputs)
        if outputs:
            all_arrays.extend(outputs)

        for arr in all_arrays:
            if hasattr(arr, "ptr") and arr.ptr:
                self.track_array(arr)

        # Track unique modules (for metadata)
        if module_hash not in self.modules:
            self.modules[module_hash] = ModuleInfo(
                module_name=module.name,
                module_hash=module_hash,
                target_arch=self.device.arch,
                cubin_filename=f"{module.name}_{module_hash_short}.cubin",
            )

        # Track kernel info (for metadata)
        if kernel.key not in self.kernels:
            hooks = launch.hooks
            mangled_name = kernel.get_mangled_name()
            self.kernels[kernel.key] = KernelInfo(
                kernel_key=kernel.key,
                module_hash=module_hash,
                forward_name=f"{mangled_name}_cuda_kernel_forward",
                backward_name=(f"{mangled_name}_cuda_kernel_backward" if hooks.backward else None),
                forward_smem_bytes=hooks.forward_smem_bytes,
                backward_smem_bytes=hooks.backward_smem_bytes if hooks.backward else 0,
                block_dim=launch.block_dim,
            )

        # Build parameter bindings
        param_bindings = self._build_param_bindings(launch, inputs, outputs)

        # Build APICLaunchInfo struct
        # Store strings as instance attributes to keep them alive
        self._kernel_key_bytes = kernel.key.encode("utf-8")
        self._module_hash_bytes = module_hash.encode("utf-8")

        info = APICLaunchInfo()
        info.kernel_key = self._kernel_key_bytes
        info.module_hash = self._module_hash_bytes

        bounds = launch.bounds
        for i in range(min(bounds.ndim, APIC_LAUNCH_MAX_DIMS)):
            info.shape[i] = bounds.shape[i]
        for i in range(bounds.ndim, APIC_LAUNCH_MAX_DIMS):
            info.shape[i] = 1

        info.ndim = bounds.ndim
        info.block_dim = launch.block_dim
        info.smem_bytes = launch.hooks.forward_smem_bytes if not launch.adjoint else launch.hooks.backward_smem_bytes
        info.is_forward = 1 if not launch.adjoint else 0

        if param_bindings:
            # Store param_bindings as instance attribute to keep it alive
            self._current_param_bindings = param_bindings
            info.params = ctypes.cast(param_bindings, ctypes.POINTER(APICParamBindingInfo))
            info.num_params = len(param_bindings)
        else:
            info.params = None
            info.num_params = 0

        return info

    def record_launch(self, launch, inputs=None, outputs=None):
        """Record a kernel launch (deprecated - use build_launch_info instead).

        This method exists for backward compatibility. New code should use
        build_launch_info() and pass the result to wp_cuda_launch_kernel().
        """
        # Just build the info - recording happens in native code during launch
        self.build_launch_info(launch, inputs, outputs)

    def _build_param_bindings(self, launch, inputs=None, outputs=None):
        """Build parameter bindings array for native code."""
        import warp

        original_arrays = []
        if inputs:
            original_arrays.extend(inputs)
        if outputs:
            original_arrays.extend(outputs)

        kernel = launch.kernel
        bindings = []

        array_idx = 0
        for i, arg in enumerate(kernel.adj.args):
            param_idx = i + 1  # Skip bounds at index 0
            param = launch.params[param_idx]
            arg_type = arg.type

            binding = APICParamBindingInfo()

            if isinstance(arg_type, warp._src.types.array):
                arr = None
                if array_idx < len(original_arrays):
                    candidate = original_arrays[array_idx]
                    if hasattr(candidate, "ptr"):
                        arr = candidate
                    array_idx += 1

                binding.type = APIC_PARAM_ARRAY
                binding.param_index = param_idx

                if arr is not None and arr.ptr:
                    region_id, offset = self.track_array(arr)
                    binding.region_id = region_id
                    binding.byte_offset = offset
                    binding.ndim = arr.ndim
                    binding.element_size = self._get_element_size(arr)
                    for d in range(min(arr.ndim, APIC_MAX_DIMS)):
                        binding.shape[d] = arr.shape[d]
                        binding.strides[d] = arr.strides[d]
                else:
                    binding.region_id = -1
                    binding.byte_offset = 0
                    binding.ndim = 0
                    binding.element_size = 0
            else:
                binding.type = APIC_PARAM_SCALAR
                binding.param_index = param_idx
                value_bytes = bytes(param)
                binding.scalar_size = len(value_bytes)
                for k, b in enumerate(value_bytes):
                    if k < APIC_MAX_SCALAR_SIZE:
                        binding.scalar_value[k] = b

            bindings.append(binding)

        if not bindings:
            return None

        # Convert to ctypes array
        arr = (APICParamBindingInfo * len(bindings))(*bindings)
        return arr

    def set_input_binding(self, name: str, arr):
        """Mark an array as an input binding."""
        region_id, offset = self.track_array(arr, MemoryRole.INPUT)
        if offset != 0:
            raise ValueError(f"Input binding '{name}' must be a base array, not a slice")
        self.input_bindings[name] = region_id

    def set_output_binding(self, name: str, arr):
        """Mark an array as an output binding."""
        region_id, offset = self.track_array(arr, MemoryRole.OUTPUT)
        if offset != 0:
            raise ValueError(f"Output binding '{name}' must be a base array, not a slice")
        self.output_bindings[name] = region_id
