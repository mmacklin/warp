# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
APIC type constants - mirrors apic_types.h constants used by Python code.

Most ctypes structures are defined in serialize.py where they're used.
"""

# =============================================================================
# Constants (must match apic_types.h)
# =============================================================================

APIC_FORMAT_VERSION = 2
APIC_MAGIC = b"WGF1"
APIC_MAX_DIMS = 4
APIC_MAX_SCALAR_SIZE = 128
APIC_LAUNCH_MAX_DIMS = 4

# Operation types
APIC_OP_KERNEL_LAUNCH = 1
APIC_OP_MEMCPY_H2D = 2
APIC_OP_MEMCPY_D2H = 3
APIC_OP_MEMCPY_D2D = 4
APIC_OP_MEMSET = 5
APIC_OP_ALLOC = 6

# Parameter binding types
APIC_PARAM_ARRAY = 1
APIC_PARAM_SCALAR = 2

# Memory region roles
APIC_ROLE_INTERNAL = 0
APIC_ROLE_INPUT = 1
APIC_ROLE_OUTPUT = 2
APIC_ROLE_INPUT_OUTPUT = 3

# Section types
APIC_SECTION_METADATA = 0x01
APIC_SECTION_MEMORY = 0x02
APIC_SECTION_OPERATIONS = 0x03
