"""Community components: add one file to this folder and it is registered.

Every module here whose name does not start with an underscore is imported by ``load_all()``, so a contribution is a
single ``<your_component>.py`` file (plus a test). Start from ``_template.py``. See ``docs/WRITING_A_COMPONENT.md``.
"""
from __future__ import annotations

import importlib
import pkgutil


def load() -> None:
    for mod in sorted(m.name for m in pkgutil.iter_modules(__path__)):
        if not mod.startswith("_"):
            importlib.import_module(f"{__name__}.{mod}")
