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
WGF (Warp Graph File) format handling.

File Structure:
    Header (64 bytes)
    Section Table
    Metadata Section (JSON)
    Memory Section
    Operations Section
"""

import json
import struct
from dataclasses import dataclass

# Magic number: "WGF1" in little-endian
WGF_MAGIC = b"WGF1"
WGF_VERSION = 2  # Version 2 uses ctypes struct-based serialization

# Section types
SECTION_METADATA = 0x01
SECTION_MEMORY = 0x02
SECTION_OPERATIONS = 0x03

# Header size
HEADER_SIZE = 64


@dataclass
class SectionEntry:
    """Section table entry."""

    type: int
    flags: int
    offset: int
    size: int
    uncompressed_size: int


class WGFWriter:
    """Writes a .wgf file."""

    def __init__(self, path: str, target_arch: int):
        self.path = path
        self.target_arch = target_arch
        self.sections: list[tuple[int, bytes]] = []  # (type, data)

    def add_metadata(self, metadata: dict):
        """Add metadata section (JSON)."""
        data = json.dumps(metadata, indent=2).encode("utf-8")
        self.sections.append((SECTION_METADATA, data))

    def add_memory(self, memory_data: bytes):
        """Add memory section."""
        self.sections.append((SECTION_MEMORY, memory_data))

    def add_operations(self, operations_data: bytes):
        """Add operations section."""
        self.sections.append((SECTION_OPERATIONS, operations_data))

    def write(self):
        """Write the .wgf file."""
        with open(self.path, "wb") as f:
            # Calculate offsets
            num_sections = len(self.sections)
            section_table_offset = HEADER_SIZE
            section_table_size = num_sections * 32  # 32 bytes per entry

            data_offset = section_table_offset + section_table_size

            # Build section entries
            entries = []
            current_offset = data_offset
            for section_type, data in self.sections:
                entry = SectionEntry(
                    type=section_type,
                    flags=0,
                    offset=current_offset,
                    size=len(data),
                    uncompressed_size=len(data),
                )
                entries.append(entry)
                current_offset += len(data)

            # Write header
            header = bytearray(HEADER_SIZE)
            header[0:4] = WGF_MAGIC
            struct.pack_into("<I", header, 4, WGF_VERSION)
            struct.pack_into("<I", header, 8, 0)  # flags
            struct.pack_into("<I", header, 12, num_sections)
            struct.pack_into("<Q", header, 16, section_table_offset)
            struct.pack_into("<I", header, 24, self.target_arch)
            # Reserved bytes 28-63 are zero
            f.write(header)

            # Write section table
            for entry in entries:
                f.write(
                    struct.pack(
                        "<IIQqq",
                        entry.type,
                        entry.flags,
                        entry.offset,
                        entry.size,
                        entry.uncompressed_size,
                    )
                )

            # Write section data
            for _, data in self.sections:
                f.write(data)


class WGFReader:
    """Reads a .wgf file."""

    def __init__(self, path: str):
        self.path = path
        self.version = 0
        self.flags = 0
        self.target_arch = 0
        self.sections: dict[int, bytes] = {}

        self._read()

    def _read(self):
        """Read and parse the .wgf file."""
        with open(self.path, "rb") as f:
            # Read header
            header = f.read(HEADER_SIZE)
            if len(header) < HEADER_SIZE:
                raise ValueError("Invalid WGF file: header too short")

            magic = header[0:4]
            if magic != WGF_MAGIC:
                raise ValueError(f"Invalid WGF file: bad magic {magic!r}")

            self.version = struct.unpack_from("<I", header, 4)[0]
            if self.version > WGF_VERSION:
                raise ValueError(f"Unsupported WGF version: {self.version}")

            self.flags = struct.unpack_from("<I", header, 8)[0]
            num_sections = struct.unpack_from("<I", header, 12)[0]
            section_table_offset = struct.unpack_from("<Q", header, 16)[0]
            self.target_arch = struct.unpack_from("<I", header, 24)[0]

            # Read section table
            f.seek(section_table_offset)
            entries = []
            for _ in range(num_sections):
                entry_data = f.read(32)
                if len(entry_data) < 32:
                    raise ValueError("Invalid WGF file: section table truncated")
                (
                    section_type,
                    flags,
                    offset,
                    size,
                    uncompressed_size,
                ) = struct.unpack("<IIQqq", entry_data[:32])
                entries.append(
                    SectionEntry(
                        type=section_type,
                        flags=flags,
                        offset=offset,
                        size=size,
                        uncompressed_size=uncompressed_size,
                    )
                )

            # Read section data
            for entry in entries:
                f.seek(entry.offset)
                data = f.read(entry.size)
                if len(data) < entry.size:
                    raise ValueError(f"Invalid WGF file: section {entry.type} truncated")
                self.sections[entry.type] = data

    def get_metadata(self) -> dict:
        """Get metadata section as dict."""
        if SECTION_METADATA not in self.sections:
            return {}
        return json.loads(self.sections[SECTION_METADATA].decode("utf-8"))

    def get_memory(self) -> bytes:
        """Get memory section."""
        return self.sections.get(SECTION_MEMORY, b"")

    def get_operations(self) -> bytes:
        """Get operations section."""
        return self.sections.get(SECTION_OPERATIONS, b"")
