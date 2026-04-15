# src/mlp_qmmm/a_input_types/__init__.py
"""
Input adapters for mlp_qmmm (v1.x)

Importing this package registers all bundled adapters with the adapter
registry in a2_parser, so they can be referenced by name in configs:

    adapter: charmm_mndo97
    adapter: numpy_folder

Each adapter module self-registers via register_adapter() at import time.
Importing this __init__ triggers all of them at once.

Registered names
----------------
  "charmm_mndo97"   — CHARMM MNDO97 MTS log files  (also: "charmmmndo97mts")
  "numpy_folder"    — folder of .npy / .npz files
"""

from mlp_qmmm.a_input_types import charmm_mndo97  # noqa: F401  registers "charmm_mndo97"
from mlp_qmmm.a_input_types import numpy_folder   # noqa: F401  registers "numpy_folder"