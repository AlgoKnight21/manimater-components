"""Text helpers: maths labels that survive a missing LaTeX install, and short annotations."""
from __future__ import annotations

from manim import DOWN, UP, Arrow, MathTex, Mobject, Text, VGroup

from ...ctx import HAS_LATEX, label
from ...tex import normalize_tex, tex_to_plain  # noqa: F401 - tex_to_plain is re-exported for viz callers
from .palette import PALETTE

class MathLabel(VGroup):
    """Intuition: a formula as a caption. MathTex when LaTeX exists, readable plain text when it does not."""

    def __init__(self, tex: str, fallback: str | None = None, *, size: int = 28, color: str = PALETTE.fg,
                 use_latex: bool | None = None):
        super().__init__()
        self.tex = tex = normalize_tex(tex)
        self.used_latex = HAS_LATEX if use_latex is None else use_latex
        if self.used_latex:
            self.add(MathTex(tex, font_size=size, color=color))
        else:
            self.add(Text(fallback or tex_to_plain(tex), font_size=size, color=color))


class Annotation(VGroup):
    """Intuition: a margin note with an arrow, so words never sit on top of the data."""

    def __init__(self, text: str, target: Mobject, *, direction=UP, arrow: bool = True, size: int = 22,
                 color: str = PALETTE.muted, gap: float = 0.7):
        super().__init__()
        self.note = label(text, size, color, width=28)
        self.note.next_to(target, direction, buff=gap)
        self.add(self.note)
        self.arrow = None
        if arrow:
            self.arrow = Arrow(self.note.get_edge_center(-direction), target.get_edge_center(direction),
                               buff=0.08, stroke_width=3, color=color, max_tip_length_to_length_ratio=0.3)
            self.add(self.arrow)
