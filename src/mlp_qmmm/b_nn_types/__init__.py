# src/mlp_qmmm/b_nn_types/__init__.py
"""
Neural network model architectures for mlp_qmmm (v1.x)

Importing this package registers all bundled models with the model registry
in b_nn_loader, so they can be referenced by name in configs:

    model:
      name: dpmm_v1

Each model module self-registers via register_model() at import time.
Importing this __init__ triggers all of them at once.

Registered names
----------------
  "dpmm_v1"   — DeepPot-MM v1: type-conditioned DeePMD + MM electrostatics
"""
from mlp_qmmm.b_nn_types import dpmm_v0  # noqa: F401  registers "dpmm_v0"
from mlp_qmmm.b_nn_types import dpmm_v1  # noqa: F401  registers "dpmm_v1"