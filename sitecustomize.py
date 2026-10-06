from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _load_stdlib_code_module() -> object:
    stdlib_path = Path(sys.base_prefix) / "Lib" / "code.py"
    if not stdlib_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("_stdlib_code", stdlib_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if any("pytest" in arg.lower() for arg in sys.argv):
    stdlib_code = _load_stdlib_code_module()
    if stdlib_code is not None:
        sys.modules["code"] = stdlib_code
