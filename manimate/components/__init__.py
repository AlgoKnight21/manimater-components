"""Component packs. Import a pack module here to register its components."""
_loaded = False


def load_all() -> None:
    global _loaded
    if _loaded:
        return
    from . import general, electrical, power, learn  # noqa: F401  (registration side effects)
    from . import physics, ml, datastructures, chemistry, databases, circuits, algorithms, finance, adv_micro  # noqa: F401
    from . import adv_ml, adv_power, adv_signals, viz_core  # noqa: F401  (viz.core devices: spotlight, focus ring, block flow)
    from . import extensions  # noqa: F401  (manim-extensions, through the generic adapter in _ext.py)
    from . import community

    community.load()  # every non-underscore module in components/community/ (one file per contribution)
    _loaded = True
