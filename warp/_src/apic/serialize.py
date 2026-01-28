# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Serialization of captured CUDA graphs to WGF format.

All recording and serialization is done in C++ (apic.cu/warp.cu).
Python just builds metadata JSON and calls wp_apic_state_save().
"""

import json
import os
from pathlib import Path

from .capture import APICapture, ModuleInfo

# Constants from apic_types.h
APIC_FORMAT_VERSION = 2


def save_graph(capture: APICapture, path: str):
    """
    Save a captured graph to disk.

    Args:
        capture: The APICapture containing the native state handle
        path: Output path for the .wgf file (without extension)

    Creates:
        - {path}.wgf: The main graph file
        - {path}_modules/: Directory containing .cubin files
    """
    import warp._src.context

    if capture.native_state is None:
        raise RuntimeError("No native APIC state - was capture started with wp_apic_begin_recording?")

    base_path = Path(path)
    wgf_path = base_path.with_suffix(".wgf")
    modules_dir = base_path.parent / f"{base_path.stem}_modules"

    # Create modules directory
    modules_dir.mkdir(parents=True, exist_ok=True)

    # Export cubin files for each unique module
    for _module_hash, module_info in capture.modules.items():
        cubin_path = modules_dir / module_info.cubin_filename
        _export_module_cubin(module_info, cubin_path)

    # Build metadata JSON
    metadata = _build_metadata(capture)
    metadata_json = json.dumps(metadata, indent=2).encode("utf-8")

    # Call native to save the graph (serializes from native state)
    runtime = warp._src.context.runtime
    result = runtime.core.wp_apic_state_save(
        capture.native_state,
        str(wgf_path).encode("utf-8"),
        capture.device.arch,
        metadata_json,
        len(metadata_json),
    )

    if not result:
        raise RuntimeError(f"Failed to write WGF file: {wgf_path}")


def _export_module_cubin(module_info: ModuleInfo, cubin_path: Path):
    """Export a module's cubin file."""
    import glob
    import shutil

    import warp
    import warp._src.context

    user_modules = warp._src.context.user_modules
    module_hash_hex = module_info.module_hash
    module_hash_short = module_hash_hex[:7]

    for module in user_modules.values():
        if module.name == module_info.module_name:
            module_name_short = module.get_module_identifier()
            arch = module_info.target_arch
            module_dir = os.path.join(warp.config.kernel_cache_dir, module_name_short)

            if os.path.exists(module_dir):
                patterns = [
                    os.path.join(module_dir, f"*.sm{arch}.cubin"),
                    os.path.join(module_dir, f"*.sm{arch}.ptx"),
                ]
                for pattern in patterns:
                    matches = glob.glob(pattern)
                    if matches:
                        shutil.copy2(matches[0], cubin_path)
                        return

            cache_dir = warp.config.kernel_cache_dir
            for root, _dirs, files in os.walk(cache_dir):
                for f in files:
                    if module_hash_short in f and (f.endswith(f".sm{arch}.cubin") or f.endswith(f".sm{arch}.ptx")):
                        shutil.copy2(os.path.join(root, f), cubin_path)
                        return

    raise ValueError(f"Could not find cubin for module {module_info.module_name} ({module_info.module_hash})")


def _build_metadata(capture: APICapture) -> dict:
    """Build the metadata dictionary."""
    modules = {
        h: {"name": i.module_name, "cubin_filename": i.cubin_filename, "target_arch": i.target_arch}
        for h, i in capture.modules.items()
    }
    kernels = {
        k: {
            "module_hash": i.module_hash,
            "forward_name": i.forward_name,
            "backward_name": i.backward_name,
            "forward_smem_bytes": i.forward_smem_bytes,
            "backward_smem_bytes": i.backward_smem_bytes,
            "block_dim": i.block_dim,
        }
        for k, i in capture.kernels.items()
    }

    return {
        "version": APIC_FORMAT_VERSION,
        "target_arch": capture.device.arch,
        "modules": modules,
        "kernels": kernels,
        "input_bindings": dict(capture.input_bindings),
        "output_bindings": dict(capture.output_bindings),
    }
