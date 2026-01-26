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
APIC (API Capture) - CUDA Graph Capture, Serialization, and Replay for Warp

This module provides functionality to:
1. Capture Warp kernel launches and memory operations
2. Serialize captured graphs to disk (.wgf format)
3. Load and execute serialized graphs
4. Generate C++ headers for native application embedding
"""

from .capture import APICapture, KernelInfo, MemoryRegion, MemoryRole, ModuleInfo
from .format import WGFReader, WGFWriter
from .serialize import LoadedGraph, load_graph, save_graph

__all__ = [
    "APICapture",
    "KernelInfo",
    "LoadedGraph",
    "MemoryRegion",
    "MemoryRole",
    "ModuleInfo",
    "WGFReader",
    "WGFWriter",
    "load_graph",
    "save_graph",
]
