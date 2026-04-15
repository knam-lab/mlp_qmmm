# src/mlp_qmmm/__init__.py
"""
mlp_qmmm — Machine-learning potentials for QM/MM simulations (v1.x)

Public API
----------
Parsing
    from mlp_qmmm.a2_parser import parse_dataset, get_adapter, register_adapter

Post-processing
    from mlp_qmmm.a1_postprocess import run_postprocess_pipeline

Model loading
    from mlp_qmmm.b_nn_loader import load_model, register_model, get_model

Training
    from mlp_qmmm.training import run

Testing
    from mlp_qmmm.testing import main as run_testing

Key / structure constants
    from mlp_qmmm.a0_structure import Keys, KEYS_POOL, KEY_POOL

Adapters and models are self-registering — importing their modules is enough:
    import mlp_qmmm.a_input_types.charmm_mndo97   # registers "charmm_mndo97"
    import mlp_qmmm.a_input_types.numpy_folder    # registers "numpy_folder"
    import mlp_qmmm.b_nn_types.dpmm_v1            # registers "dpmm_v1"
    import mlp_qmmm.b_nn_types.dpmm_v2            # registers "dpmm_v2"

Alternatively, import the subpackages directly and all adapters/models in
those packages are registered automatically:
    import mlp_qmmm.a_input_types
    import mlp_qmmm.b_nn_types
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mlp_qmmm")
except PackageNotFoundError:
    __version__ = "dev"

__all__ = ["__version__"]