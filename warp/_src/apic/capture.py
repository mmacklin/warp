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
"""

import ctypes
from dataclasses import dataclass, field


# Memory region roles (must match C++ APICMemoryRole enum)
class MemoryRole:
    INTERNAL = 0
    INPUT = 1
    OUTPUT = 2
    INPUT_OUTPUT = 3


@dataclass
class MemoryRegion:
    """Represents a contiguous memory allocation."""

    region_id: int
    size: int  # Size in bytes
    device_ptr: int  # Original device pointer
    element_size: int  # Size of one element in bytes
    role: int = MemoryRole.INTERNAL
    initial_data: bytes | None = None  # For internal arrays: serialized content

    def update_role(self, new_role: int):
        """Update role, preferring higher priority roles."""
        # INPUT_OUTPUT > OUTPUT > INPUT > INTERNAL
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


@dataclass
class LaunchRecord:
    """Records a kernel launch with all necessary context."""

    kernel_key: str
    module_hash: str
    dim: int
    max_blocks: int
    block_dim: int
    smem_bytes: int
    is_forward: bool
    # Parameter bindings: list of (param_index, region_id, offset, shape, strides, element_size)
    # For scalars: (param_index, None, 0, None, None, size, value_bytes)
    param_bindings: list = field(default_factory=list)


@dataclass
class MemcpyRecord:
    """Records a memory copy operation."""

    dst_region_id: int
    dst_offset: int
    src_region_id: int | None  # None for host source
    src_offset: int
    size: int
    kind: str  # "H2D", "D2H", "D2D"
    # For H2D: store the source data
    src_data: bytes | None = None


@dataclass
class MemsetRecord:
    """Records a memory set operation."""

    region_id: int
    offset: int
    value: int
    size: int


class APICapture:
    """Records API calls during CUDA graph capture for serialization."""

    def __init__(self, device, stream=None):
        self.device = device
        self.stream = stream or device.stream

        # Recorded data
        self.launches: list[LaunchRecord] = []
        self.memory_ops: list = []  # MemcpyRecord or MemsetRecord
        self.operations: list = []  # All operations in order: ("launch", idx) or ("memop", idx)
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
        # APIC recording is done at the Python level via record_launch()
        # No C++ state needed for the initial implementation
        self._recording = True

    def end(self):
        """End APIC recording."""
        self._recording = False

    def is_recording(self) -> bool:
        """Check if recording is active."""
        return getattr(self, "_recording", False)

    def track_array(self, arr, role: int = MemoryRole.INTERNAL) -> tuple[int, int]:
        """
        Track an array, resolving to its base allocation.

        Returns:
            (region_id, offset) - The region ID and byte offset within the region
        """
        # Walk the _ref chain to find base allocation
        base = arr
        offset = 0
        while hasattr(base, "_ref") and base._ref is not None:
            # Calculate offset from parent
            offset += base.ptr - base._ref.ptr
            base = base._ref

        base_ptr = base.ptr
        base_size = base.capacity

        # Get element size
        element_size = self._get_element_size(arr)

        # Check if we already have this region
        if base_ptr in self.memory_regions:
            region = self.memory_regions[base_ptr]
            region.update_role(role)
            region_id = region.region_id
        else:
            # Create new region
            region_id = self._next_region_id
            self._next_region_id += 1

            region = MemoryRegion(
                region_id=region_id,
                size=base_size,
                device_ptr=base_ptr,
                element_size=element_size,
                role=role,
                initial_data=None,
            )
            self.memory_regions[base_ptr] = region

        # Map this array's ptr to the region
        self._ptr_to_region_id[arr.ptr] = region_id

        return region_id, offset

    def _get_element_size(self, arr) -> int:
        """Get the size of one element in bytes."""
        dtype = arr.dtype
        if hasattr(dtype, "_type_size_"):
            return dtype._type_size_
        elif hasattr(dtype, "_length_") and hasattr(dtype, "_type_"):
            # Vector/matrix type
            return dtype._length_ * ctypes.sizeof(dtype._type_)
        else:
            return ctypes.sizeof(dtype)

    def record_launch(self, launch, inputs=None, outputs=None):
        """Record a kernel launch from a Launch object.

        Args:
            launch: The Launch object containing kernel and parameter info
            inputs: Original input arrays (optional, for memory region tracking)
            outputs: Original output arrays (optional, for memory region tracking)
        """
        kernel = launch.kernel
        module = kernel.module
        module_exec = launch.module_exec
        # Convert bytes hash to hex string for serialization
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

        # Track unique modules
        if module_hash not in self.modules:
            self.modules[module_hash] = ModuleInfo(
                module_name=module.name,
                module_hash=module_hash,
                target_arch=self.device.arch,
                cubin_filename=f"{module.name}_{module_hash_short}.cubin",
            )

        # Track kernel info
        if kernel.key not in self.kernels:
            hooks = launch.hooks
            # Get the kernel's mangled name (includes kernel-specific hash)
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

        # Record the launch
        record = LaunchRecord(
            kernel_key=kernel.key,
            module_hash=module_hash,
            dim=launch.bounds.size,
            max_blocks=launch.max_blocks,
            block_dim=launch.block_dim,
            smem_bytes=(launch.hooks.backward_smem_bytes if launch.adjoint else launch.hooks.forward_smem_bytes),
            is_forward=not launch.adjoint,
            param_bindings=[],
        )

        # Record parameter bindings using original arrays
        self._record_param_bindings(record, launch, inputs, outputs)

        self.launches.append(record)
        self.operations.append(("launch", len(self.launches) - 1))

    def _record_param_bindings(self, record: LaunchRecord, launch, inputs=None, outputs=None):
        """Extract parameter bindings from a Launch object.

        Args:
            record: The LaunchRecord to populate
            launch: The Launch object containing kernel and parameter info
            inputs: Original input arrays passed to wp.launch
            outputs: Original output arrays passed to wp.launch (optional)
        """
        import warp

        # Build a list of original arrays from inputs/outputs
        # The kernel arguments come from inputs first, then outputs (if separate)
        original_arrays = []
        if inputs:
            original_arrays.extend(inputs)
        if outputs:
            original_arrays.extend(outputs)

        # params[0] is bounds, params[1:] are kernel arguments
        kernel = launch.kernel

        array_idx = 0  # Index into original_arrays for array params
        for i, arg in enumerate(kernel.adj.args):
            param_idx = i + 1  # Skip bounds at index 0
            param = launch.params[param_idx]
            arg_type = arg.type

            if isinstance(arg_type, warp._src.types.array):
                # Array parameter - use original array from inputs/outputs
                arr = None
                if array_idx < len(original_arrays):
                    candidate = original_arrays[array_idx]
                    if hasattr(candidate, "ptr"):
                        arr = candidate
                    array_idx += 1

                if arr is not None and arr.ptr:
                    region_id, offset = self.track_array(arr)
                    record.param_bindings.append(
                        {
                            "type": "array",
                            "param_index": param_idx,
                            "region_id": region_id,
                            "offset": offset,
                            "shape": list(arr.shape),
                            "strides": list(arr.strides),
                            "ndim": arr.ndim,
                            "element_size": self._get_element_size(arr),
                        }
                    )
                else:
                    # Null array
                    record.param_bindings.append(
                        {
                            "type": "array",
                            "param_index": param_idx,
                            "region_id": None,
                            "offset": 0,
                            "shape": [],
                            "strides": [],
                            "ndim": 0,
                            "element_size": 0,
                        }
                    )
            else:
                # Scalar parameter - serialize value
                value_bytes = bytes(param)
                record.param_bindings.append(
                    {
                        "type": "scalar",
                        "param_index": param_idx,
                        "size": len(value_bytes),
                        "value": value_bytes,
                    }
                )

    def finalize_memory_data(self):
        """Capture initial data for internal memory regions."""
        import warp

        for base_ptr, region in self.memory_regions.items():
            if region.role == MemoryRole.INTERNAL:
                # Copy data from device to host
                data = (ctypes.c_uint8 * region.size)()
                warp._src.context.runtime.core.wp_memcpy_d2h(
                    self.device.context,
                    ctypes.cast(data, ctypes.c_void_p),
                    ctypes.c_void_p(base_ptr),
                    region.size,
                    None,  # Use current stream
                )
                # Synchronize to ensure copy is complete
                warp.synchronize_device(self.device)
                region.initial_data = bytes(data)

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

    def record_memcpy_d2d(self, dest, dest_offset: int, src, src_offset: int, count: int):
        """Record a device-to-device memory copy operation.

        Args:
            dest: Destination array
            dest_offset: Element offset in destination
            src: Source array
            src_offset: Element offset in source
            count: Number of elements to copy
        """
        import warp

        # Track both arrays
        dst_region_id, dst_byte_offset = self.track_array(dest)
        src_region_id, src_byte_offset = self.track_array(src)

        # Calculate byte offsets including element offsets
        element_size = warp._src.types.type_size_in_bytes(src.dtype)
        dst_byte_offset += dest_offset * element_size
        src_byte_offset += src_offset * element_size
        size = count * element_size

        record = MemcpyRecord(
            dst_region_id=dst_region_id,
            dst_offset=dst_byte_offset,
            src_region_id=src_region_id,
            src_offset=src_byte_offset,
            size=size,
            kind="D2D",
            src_data=None,
        )
        self.memory_ops.append(record)
        self.operations.append(("memop", len(self.memory_ops) - 1))
