"""Which components wrap a community plugin, and that the credit shown on the website is accurate.

These need none of the plugins installed: they only read the registry, the catalog and docs/THIRD_PARTY.md.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from manimate import registry
from manimate.catalog import build_catalog
from manimate.components import load_all
from manimate.plugins import PLUGINS

load_all()

PLUGIN_MODULES = {"physics", "ml", "datastructures", "chemistry", "databases", "circuits", "_ext"}


def _third_party() -> str:
    """docs/THIRD_PARTY.md sits at the repo root in both repos (manimater and manimater-components), at different depths."""
    for parent in Path(__file__).resolve().parents:
        doc = parent / "docs" / "THIRD_PARTY.md"
        if doc.exists():
            return doc.read_text(encoding="utf-8")
    raise FileNotFoundError("docs/THIRD_PARTY.md")


THIRD_PARTY = _third_party()


def _module(comp) -> str:
    return comp.fn.__module__.rsplit(".", 1)[-1]


def test_components_in_plugin_packs_name_their_plugin_and_others_do_not():
    for comp in registry.all_components():
        if _module(comp) in PLUGIN_MODULES:
            assert comp.plugin in PLUGINS, f"{comp.name} wraps a community plugin but does not say which"
        else:
            assert comp.plugin is None, f"{comp.name} is not in a plugin pack but names {comp.plugin}"


def test_every_plugin_has_a_component_and_every_component_a_real_plugin():
    used = {c.plugin for c in registry.all_components() if c.plugin}
    assert used == set(PLUGINS), f"unused: {set(PLUGINS) - used}, unknown: {used - set(PLUGINS)}"


def test_catalog_carries_the_plugin_credit():
    for entry in build_catalog():
        comp = registry.get(entry["name"])
        if comp.plugin:
            assert entry["plugin"] == {"name": comp.plugin, "url": PLUGINS[comp.plugin].url,
                                       "license": PLUGINS[comp.plugin].license}
        else:
            assert "plugin" not in entry


def test_plugin_table_matches_the_licence_document():
    used_section = THIRD_PARTY.split("## Used", 1)[1].split("##", 1)[0]
    for plugin in PLUGINS.values():
        row = next((line for line in used_section.splitlines() if line.startswith("|") and plugin.url in line), None)
        assert row, f"{plugin.name} is not in the 'Used' table of docs/THIRD_PARTY.md"
        assert re.search(r"\bMIT\b", row), f"{plugin.name} is not listed as MIT in docs/THIRD_PARTY.md"
        assert plugin.license.startswith("MIT")
        assert plugin.url.startswith("https://github.com/")


def test_the_extra_licence_condition_is_visible_on_the_credit():
    # manim-eng's author adds a no-AI-training clause; a bare "MIT" on the website would hide it
    assert "AI" in PLUGINS["manim-eng"].license
    assert all(p.license == "MIT" for n, p in PLUGINS.items() if n != "manim-eng")


def test_unknown_plugin_is_rejected_at_registration():
    with pytest.raises(ValueError, match="unknown plugin"):
        registry.component("x_test_component", "d", {}, {}, plugin="not-a-plugin")
