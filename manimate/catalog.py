"""Export the component catalog: ``python -m manimate.catalog > catalog.json``."""
from __future__ import annotations

import json

from . import registry
from .components import load_all


def build_catalog() -> list[dict]:
    load_all()
    return [c.to_catalog() for c in registry.all_components()]


if __name__ == "__main__":
    print(json.dumps(build_catalog(), indent=2, ensure_ascii=False))
