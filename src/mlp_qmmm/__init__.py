"""Top-level package for mlp_qmmm."""  
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("mlp_qmmm")
except PackageNotFoundError:
    __version__ = "dev"
