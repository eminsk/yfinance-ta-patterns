"""Automatic compatibility hook for Python 3.8 / PyPy 3.8 environments.

Intercepts modules (such as yfinance) that use PEP 604 union syntax ('|') in
annotations without 'from __future__ import annotations', and automatically
injects future annotations at runtime so they import cleanly without TypeError.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys

_INSTALLED = False


class _FutureAnnotationsPatchLoader(importlib.abc.Loader):
    def __init__(self, original_loader):
        self.original_loader = original_loader

    def create_module(self, spec):
        if hasattr(self.original_loader, "create_module"):
            return self.original_loader.create_module(spec)
        return None

    def exec_module(self, module):
        fullname = getattr(module, "__name__", "")
        origin = getattr(module, "__file__", None)
        try:
            source = None
            if hasattr(self.original_loader, "get_source"):
                source = self.original_loader.get_source(fullname)
            if source is not None:
                if "from __future__ import annotations" not in source:
                    source = "from __future__ import annotations\n" + source
                code = compile(source, origin or "<string>", "exec")
                exec(code, module.__dict__)
                return
        except Exception:
            pass

        # Fallback to original loader
        self.original_loader.exec_module(module)


class _CompatMetaPathFinder(importlib.abc.MetaPathFinder):
    # Target modules known to contain PEP 604 annotations breaking Python 3.8
    TARGET_PREFIXES = ("yfinance",)

    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith(self.TARGET_PREFIXES):
            return None

        for finder in sys.meta_path:
            if finder is self:
                continue
            if hasattr(finder, "find_spec"):
                spec = finder.find_spec(fullname, path, target)
                if spec and spec.loader and hasattr(spec.loader, "get_source"):
                    spec.loader = _FutureAnnotationsPatchLoader(spec.loader)
                    return spec
        return None


def install_compat_hook():
    """Install the import hook if running on Python < 3.9."""
    global _INSTALLED
    if _INSTALLED or sys.version_info >= (3, 9):
        return
    _INSTALLED = True
    sys.meta_path.insert(0, _CompatMetaPathFinder())
