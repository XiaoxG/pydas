"""PyDAS core subpackage: the ``PyDAS`` facade and mixin proxies.

Internal helpers live in ``state``, ``channels``, and ``io_format``.
Public method names on :class:`PyDAS` stay frozen; new internal functions
use snake_case.
"""
from .pydas_obj import PyDAS

__all__ = ["PyDAS"]
