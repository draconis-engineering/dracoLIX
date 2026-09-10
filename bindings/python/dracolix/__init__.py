"""
Python API - outgoing-only binding to dracolix_core.
No NumPy required for core; NumPy is used only as a data-exchange format
in the binding layer (not as compute backend).
"""
try:
    from ._dracolix_nb import *  # nanobind extension, built from bindings/python/src
except ImportError:
    # Core not yet built with Python bindings - pure Python fallback for docs
    pass

__version__ = "0.1.0-core"
