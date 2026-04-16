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

import warp._src.types

# Use constants from types.py
ARRAY_MAX_DIMS = warp._src.types.ARRAY_MAX_DIMS
LAUNCH_MAX_DIMS = warp._src.types.LAUNCH_MAX_DIMS


class APICLaunchParam(ctypes.Structure):
    """Parameter binding info for array or scalar parameters - matches C struct in apic_types.h.

    For arrays: uses region_id, byte_offset, shape, strides, element_size
    For scalars: is_array=0, scalar_size in byte_offset, value bytes in shape[] and strides[]
    """

    _pack_ = 1
    _fields_ = [  # noqa: RUF012
        ("is_array", ctypes.c_uint8),  # 1 for array, 0 for scalar
        ("ndim", ctypes.c_uint8),  # Number of dimensions (arrays only)
        ("param_index", ctypes.c_uint16),
        ("region_id", ctypes.c_int32),  # For arrays, -1 for null or scalar
        ("byte_offset", ctypes.c_uint64),  # Byte offset (arrays) or scalar_size (scalars)
        ("shape", ctypes.c_int64 * ARRAY_MAX_DIMS),  # Array shape or first 32 bytes of scalar
        ("strides", ctypes.c_int64 * ARRAY_MAX_DIMS),  # Array strides or next 32 bytes of scalar
        ("element_size", ctypes.c_uint32),  # Element size (arrays only)
        ("_pad1", ctypes.c_uint32),
    ]


class APICLaunchInfo(ctypes.Structure):
    """Launch info passed to wp_cuda_launch_kernel() - matches C struct in apic_types.h.

    Only includes fields needed to identify the kernel. Other launch parameters
    (dim, block_dim, smem_bytes) are passed directly to wp_cuda_launch_kernel(),
    and shape/ndim are in launch_bounds_t which is always args[0].
    """

    _pack_ = 1
    _fields_ = [  # noqa: RUF012
        ("kernel_key", ctypes.c_char_p),
        ("module_hash", ctypes.c_char_p),
        ("is_forward", ctypes.c_uint8),
        ("ndim", ctypes.c_uint8),  # Number of launch dimensions, for parsing launch_bounds_t<N>
        ("_pad", ctypes.c_uint8 * 2),
        ("params", ctypes.POINTER(APICLaunchParam)),
        ("num_params", ctypes.c_int32),
    ]


@dataclass
class MemoryRegion:
    """Represents a contiguous memory allocation."""

    region_id: int
    size: int  # Size in bytes
    device_ptr: int  # Original device pointer
    element_size: int  # Size of one element in bytes


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
        self.stream = stream if stream is not None else (device.stream if device.is_cuda else None)

        # Native APIC state handle
        self.native_state = None

        # Metadata (still tracked in Python for building metadata JSON)
        self.memory_regions: dict[int, MemoryRegion] = {}  # base_ptr -> region
        self.modules: dict[str, ModuleInfo] = {}  # module_hash -> ModuleInfo
        self.kernels: dict[str, KernelInfo] = {}  # kernel_key -> KernelInfo

        # Named bindings (set during save)
        self.bindings: dict[str, int] = {}  # name -> region_id

        # Internal tracking
        self._recording: bool = False
        self._track_memory: bool = True  # Default to full tracking; begin() may override

    def begin(self, track_memory=True):
        """Start APIC recording."""
        import warp._src.context

        self._track_memory = track_memory
        runtime = warp._src.context.runtime

        # Create native state
        self.native_state = runtime.core.wp_apic_create_state()
        if not self.native_state:
            raise RuntimeError("Failed to create APIC state")

        if self.device.is_cpu:
            runtime.core.wp_apic_set_cpu_mode(self.native_state)

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

    def _find_handle_offsets(self, dtype, base_offset: int = 0) -> list[int]:
        """Recursively find byte offsets of wp.handle fields in a type.

        Handles:
        - Direct wp.handle type
        - wp.struct containing wp.handle fields (nested recursively)
        - wp.array dtype that is a struct containing handles

        Returns list of byte offsets where handle pointers are located.
        """
        import warp
        import warp._src.codegen

        offsets = []
        if dtype is warp.handle:
            offsets.append(base_offset)
        elif isinstance(dtype, warp._src.codegen.Struct):
            # wp.struct - inspect its fields
            for field_name, var in dtype.vars.items():
                field_offset = getattr(dtype.ctype, field_name).offset
                offsets.extend(self._find_handle_offsets(var.type, base_offset + field_offset))
        return offsets

    def track_array(self, arr) -> tuple[int, int]:
        """
        Track an array, resolving to its base allocation.
        Registers the region with native code if new.
        Also automatically registers any handle pointer locations in the dtype.

        Returns:
            (region_id, offset) - The region ID and byte offset within the region
        """
        import warp
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
            return region.region_id, offset

        # Register with native code
        runtime = warp._src.context.runtime
        region_id = runtime.core.wp_apic_register_memory_region(
            self.native_state,
            base_ptr,
            base_size,
            element_size,
        )

        # Track in Python for metadata
        region = MemoryRegion(
            region_id=region_id,
            size=base_size,
            device_ptr=base_ptr,
            element_size=element_size,
        )
        self.memory_regions[base_ptr] = region

        # Auto-detect handle locations (only needed for serialization/memory tracking)
        if self._track_memory:
            handle_offsets = self._find_handle_offsets(arr.dtype)
            if handle_offsets:
                stride = warp.types.type_size_in_bytes(arr.dtype)
                for handle_offset in handle_offsets:
                    runtime.core.wp_apic_register_ptr_location(self.native_state, region_id, handle_offset, stride)

        return region_id, offset

    def _get_element_size(self, arr) -> int:
        """Get the size of one element in bytes."""
        import warp._src.codegen

        dtype = arr.dtype
        if hasattr(dtype, "_type_size_"):
            return dtype._type_size_
        elif hasattr(dtype, "_length_") and hasattr(dtype, "_type_"):
            return dtype._length_ * ctypes.sizeof(dtype._type_)
        elif isinstance(dtype, warp._src.codegen.Struct):
            return ctypes.sizeof(dtype.ctype)
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
            if self.device.is_cpu:
                forward_name = f"{mangled_name}_cpu_forward"
                backward_name = f"{mangled_name}_cpu_backward" if hooks.backward else None
            else:
                forward_name = f"{mangled_name}_cuda_kernel_forward"
                backward_name = f"{mangled_name}_cuda_kernel_backward" if hooks.backward else None
            self.kernels[kernel.key] = KernelInfo(
                kernel_key=kernel.key,
                module_hash=module_hash,
                forward_name=forward_name,
                backward_name=backward_name,
                forward_smem_bytes=hooks.forward_smem_bytes,
                backward_smem_bytes=hooks.backward_smem_bytes if hooks.backward else 0,
                block_dim=launch.block_dim,
            )

        # Build parameter bindings
        launch_params = self._build_launch_params(launch, inputs, outputs)

        # Build APICLaunchInfo struct
        # Store strings as instance attributes to keep them alive
        self._kernel_key_bytes = kernel.key.encode("utf-8")
        self._module_hash_bytes = module_hash.encode("utf-8")

        info = APICLaunchInfo()
        info.kernel_key = self._kernel_key_bytes
        info.module_hash = self._module_hash_bytes
        info.is_forward = 1 if not launch.adjoint else 0
        info.ndim = len(launch.bounds.shape)

        if launch_params:
            # Store launch_params as instance attribute to keep it alive
            self._current_launch_params = launch_params
            info.params = ctypes.cast(launch_params, ctypes.POINTER(APICLaunchParam))
            info.num_params = len(launch_params)
        else:
            info.params = None
            info.num_params = 0

        return info

    def _build_launch_params(self, launch, inputs=None, outputs=None):
        """Build launch params array for all kernel parameters.

        Launch bounds (shape/ndim/size) are embedded directly in the launch record.
        All other parameters starting from index 1 are captured.
        """
        import warp

        # Filter to only include actual arrays (items with ptr attribute)
        original_arrays = []
        if inputs:
            for item in inputs:
                if hasattr(item, "ptr"):
                    original_arrays.append(item)
        if outputs:
            for item in outputs:
                if hasattr(item, "ptr"):
                    original_arrays.append(item)

        kernel = launch.kernel
        params = []

        # Capture all parameters (starting from index 1, param 0 is launch_bounds)
        array_idx = 0
        for i, arg in enumerate(kernel.adj.args):
            param_idx = i + 1
            arg_type = arg.type

            param = APICLaunchParam()
            param.param_index = param_idx

            if isinstance(arg_type, warp._src.types.array):
                # Array parameter
                param.is_array = 1

                arr = None
                if array_idx < len(original_arrays):
                    arr = original_arrays[array_idx]
                    array_idx += 1

                if arr is not None and arr.ptr:
                    region_id, offset = self.track_array(arr)
                    param.region_id = region_id
                    param.byte_offset = offset
                    param.ndim = arr.ndim
                    param.element_size = self._get_element_size(arr)
                    for d in range(min(arr.ndim, ARRAY_MAX_DIMS)):
                        param.shape[d] = arr.shape[d]
                        param.strides[d] = arr.strides[d]
                else:
                    param.region_id = -1
                    param.byte_offset = 0
                    param.ndim = 0
                    param.element_size = 0
            else:
                # Scalar parameter - store value bytes in shape[] and strides[]
                param.is_array = 0
                param.ndim = 0
                param.region_id = -1
                param.element_size = 0

                # Get scalar bytes from the ctypes param
                value_bytes = bytes(launch.params[param_idx])
                scalar_size = len(value_bytes)
                param.byte_offset = scalar_size  # Store size in byte_offset

                # Store value bytes in shape[] (32 bytes) and strides[] (32 bytes)
                max_scalar_size = ARRAY_MAX_DIMS * 8 * 2  # 64 bytes
                if scalar_size > max_scalar_size:
                    raise ValueError(f"Scalar parameter too large: {scalar_size} bytes (max {max_scalar_size})")

                # Copy bytes into shape[] and strides[] arrays
                shape_bytes = (ctypes.c_uint8 * (ARRAY_MAX_DIMS * 8)).from_buffer_copy(
                    value_bytes[: min(scalar_size, ARRAY_MAX_DIMS * 8)].ljust(ARRAY_MAX_DIMS * 8, b"\x00")
                )
                param.shape = (ctypes.c_int64 * ARRAY_MAX_DIMS).from_buffer_copy(shape_bytes)

                if scalar_size > ARRAY_MAX_DIMS * 8:
                    strides_bytes = (ctypes.c_uint8 * (ARRAY_MAX_DIMS * 8)).from_buffer_copy(
                        value_bytes[ARRAY_MAX_DIMS * 8 :].ljust(ARRAY_MAX_DIMS * 8, b"\x00")
                    )
                    param.strides = (ctypes.c_int64 * ARRAY_MAX_DIMS).from_buffer_copy(strides_bytes)

            params.append(param)

        if not params:
            return None

        # Convert to ctypes array
        result = (APICLaunchParam * len(params))(*params)
        return result

    def set_binding(self, name: str, arr):
        """Mark an array as a named parameter."""
        region_id, offset = self.track_array(arr)
        if offset != 0:
            raise ValueError(f"Parameter '{name}' must be a base array, not a slice")
        self.bindings[name] = region_id
