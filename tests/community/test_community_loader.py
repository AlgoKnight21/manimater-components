"""The community folder: every non-underscore module is loaded, and every component in it passes the same checks."""
from __future__ import annotations

import importlib.util
import pathlib

import pytest
from manim import FadeOut, Scene, tempconfig

from manimate import registry
from manimate.components import community, load_all
from manimate.ctx import Ctx

load_all()
COMMUNITY_DIR = pathlib.Path(community.__file__).parent
COMMUNITY = [c for c in registry.all_components() if c.fn.__module__.startswith("manimate.components.community.")]


def _dry_run(comp, params, seconds: float = 6.0) -> None:
    clean, errs = registry.validate_params(comp, params)
    assert not errs, errs
    with tempconfig({"dry_run": True, "disable_caching": True, "verbosity": "ERROR", "progress_bar": "none",
                     "write_to_movie": False, "frame_rate": 4}):
        scene = Scene()
        comp.fn(Ctx(scene, seconds), **clean)
        scene.play(*[FadeOut(m) for m in scene.mobjects], run_time=0.5)


def test_underscore_modules_are_not_loaded():
    assert "template" not in {c.name for c in COMMUNITY}
    assert not any(c.fn.__module__.endswith("._template") for c in registry.all_components())


@pytest.mark.parametrize("comp", COMMUNITY, ids=lambda c: c.name)
def test_every_community_component_example_renders(comp):
    _dry_run(comp, comp.example)


@pytest.mark.parametrize("comp", COMMUNITY, ids=lambda c: c.name)
def test_community_components_keep_to_the_rules(comp):
    assert comp.name == comp.name.lower() and comp.name.replace("_", "").isalnum(), "name must be snake_case"
    assert 0 < len(comp.params) <= 6, "few params (at most 6)"
    assert len(comp.description) <= 200
    src = pathlib.Path(importlib.import_module(comp.fn.__module__).__file__).read_text()
    for banned in ("eval(", "exec(", "subprocess", "os.system", "requests", "urllib", "socket", "__import__"):
        assert banned not in src, f"{banned!r} is not allowed in a component"


def test_the_template_is_a_working_component():
    """_template.py is never imported by the library, so load it here under a throwaway registry entry."""
    spec = importlib.util.spec_from_file_location("manimate.components.community._template_under_test", COMMUNITY_DIR / "_template.py")
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "manimate.components.community"
    spec.loader.exec_module(mod)
    comp = registry.get("step_chain")
    try:
        assert comp is not None
        _dry_run(comp, comp.example)
        _dry_run(comp, {"steps": ["a", "b"]}, seconds=4.0)
        assert mod._clean_steps(["  one ", "", "two", "x" * 40])[2] == "x" * 24
    finally:
        registry._REGISTRY.pop("step_chain", None)
