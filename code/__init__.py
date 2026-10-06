"""Compatibility wrapper for the project package while preserving stdlib `code` imports."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent)]

stdlib_code_path = Path(sys.base_prefix) / "Lib" / "code.py"
if stdlib_code_path.exists():
    spec = importlib.util.spec_from_file_location("_stdlib_code", stdlib_code_path)
    if spec is not None and spec.loader is not None:
        _stdlib_code = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_stdlib_code)
        for _name in dir(_stdlib_code):
            if not _name.startswith("__"):
                globals()[_name] = getattr(_stdlib_code, _name)
        __all__ = [name for name in dir(_stdlib_code) if not name.startswith("__")]
