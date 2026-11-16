"""README section 13, made automatic: colours come from the palette, and every Mobject says what it pictures."""
import importlib
import inspect
import pkgutil
import re
from pathlib import Path

import manimate.viz.core as core
import manimate.viz.ml_advanced as ml_advanced

VIZ_DIR = Path(core.__file__).parent.parent
NOT_VISUALS = {"Palette", "HeatmapData", "VizScene", "RecordingCtx", "AttentionData", "PEComponent", "KVStep",
               "HeadData", "QKVData", "PEWavesData", "PESumData", "PEHeatmapData", "KVCacheBufferData", "DecodeTimelineData",
               "MultiHeadData"}  # data classes and test harness, not pictures
# Manim's own colour constants. `theme.YELLOW` / `theme.PURPLE` are the repo theme's tokens (manimate/theme.py), which is
# where colours are meant to come from, so a name reached through `theme.` is not a hard-coded Manim colour.
MANIM_COLOURS = re.compile(r"(?<!theme\.)\b(WHITE|BLACK|RED|GREEN|BLUE|YELLOW|ORANGE|PURPLE|TEAL|GREY|GRAY|PINK|GOLD|MAROON)\b")


def _modules():
    for pkg in (core, ml_advanced):
        for m in pkgutil.iter_modules(pkg.__path__):
            if not m.name.startswith("_"):
                yield importlib.import_module(f"{pkg.__name__}.{m.name}")


def test_no_hex_literals_or_manim_colour_constants_outside_the_palette():
    for path in VIZ_DIR.rglob("*.py"):
        if path.name in ("palette.py", "theme_bridge.py"):
            continue
        src = path.read_text()
        assert not re.search(r"#[0-9a-fA-F]{6}\b", src), f"hex literal in {path.name}"
        assert not MANIM_COLOURS.search(src), f"manim colour constant in {path.name}"


def test_every_public_visual_has_an_intuition_line():
    missing = []
    for mod in _modules():
        for name, obj in vars(mod).items():
            if name.startswith("_") or name in NOT_VISUALS or getattr(obj, "__module__", None) != mod.__name__:
                continue
            if inspect.isclass(obj) or inspect.isfunction(obj):
                if name in {"Beats"} or name[0].isupper() or name in {"pulse_along", "fade_in_whole"}:
                    if "Intuition:" not in (inspect.getdoc(obj) or "") and not name.startswith("tex_") and name != "dry_run":
                        missing.append(f"{mod.__name__}.{name}")
    assert not missing, missing
