from manim import DOWN, Square

from manimate.viz.core.labels import Annotation, MathLabel, tex_to_plain


def test_tex_to_plain_handles_fractions_greek_and_scripts():
    assert tex_to_plain(r"\frac{V_{in}}{R}") == "(V_in)/(R)"
    assert tex_to_plain(r"\sigma^{2} \cdot \sqrt{x}") == "σ^(2) · √(x)"


def test_mathlabel_falls_back_to_text_without_latex():
    m = MathLabel(r"\alpha + \beta", use_latex=False)
    assert m.used_latex is False and len(m) == 1 and m.width > 0
    assert MathLabel(r"x", fallback="ex", use_latex=False).tex == "x"


def test_annotation_sits_beside_target_and_points_at_it():
    sq = Square()
    a = Annotation("look here", sq, direction=DOWN)
    assert a.note.get_center()[1] < sq.get_center()[1]
    assert a.arrow is not None and len(Annotation("x", sq, arrow=False)) == 1
