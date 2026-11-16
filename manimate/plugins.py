"""The community Manim plugins that components wrap.

A component names its plugin in ``@component(..., plugin="manim-physics")``. The catalog then carries the plugin's name,
link and licence, and the website's Components page credits it. ``tests/test_plugin_metadata.py`` checks this table
against ``docs/THIRD_PARTY.md``, which is where the licences are explained.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Plugin:
    name: str
    url: str
    license: str

    def to_catalog(self) -> dict[str, str]:
        return {"name": self.name, "url": self.url, "license": self.license}


PLUGINS: dict[str, Plugin] = {p.name: p for p in (
    Plugin("manim-physics", "https://github.com/Matheart/manim-physics", "MIT"),
    Plugin("ManimML", "https://github.com/helblazer811/ManimML", "MIT"),
    Plugin("manim-dsa", "https://github.com/F4bbi/manim-dsa", "MIT"),
    Plugin("manim-chemistry", "https://github.com/UnMolDeQuimica/manim-chemistry", "MIT"),
    Plugin("manim-databases", "https://github.com/pg-pilot/manim-databases", "MIT"),
    # The author adds one condition to the MIT licence, so the page states it rather than showing a bare "MIT".
    Plugin("manim-extensions", "https://github.com/ExploreMaths/manim_extensions", "MIT"),
    Plugin("manim-eng", "https://github.com/overegneered/manim-eng", "MIT, no AI-model training"),
)}
