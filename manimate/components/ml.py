"""Machine-learning pack, built on the community plugin ManimML (https://github.com/helblazer811/ManimML).

ManimML 0.0.24 predates Manim 0.19+, so the calls go through ``tolerate_empty_groups`` (see ``_plugin.py``).
"""
from __future__ import annotations

from manim import DOWN, Create, FadeIn, Write

from .. import theme
from ..ctx import Ctx, label
from ..registry import Param as P, component
from ._plugin import fit, quiet, require, tolerate_empty_groups

_PIP = "manim-ml"


def _layers_ok(v: list) -> str | None:
    if not 2 <= len(v) <= 6:
        return "needs 2 to 6 layers"
    if any(not float(x).is_integer() or not 1 <= x <= 8 for x in v):
        return "each layer size must be a whole number from 1 to 8"
    return None


@component(
    "neural_network",
    "Feed-forward neural network, layer by layer, with an optional forward pass.",
    {"layers": P("number[]", "neurons per layer, 2-6 layers of 1-8, e.g. [3, 5, 4, 2]", validator=_layers_ok),
     "forward_pass": P("boolean", "animate data flowing through", default=True)},
    {"layers": [3, 5, 4, 2], "forward_pass": True},
    domain="machine-learning",
    plugin="ManimML",
)
def neural_network(ctx: Ctx, layers: list[float] | None = None, forward_pass: bool = True) -> None:
    ml = require("manim_ml.neural_network", _PIP)
    sizes = [int(n) for n in (layers or [3, 5, 4, 2])]
    head = ctx.heading("Neural network")
    with quiet(), tolerate_empty_groups():
        net = ml.NeuralNetwork([ml.FeedForwardLayer(n) for n in sizes], layer_spacing=0.9)
        fit(net, max_w=11.5, max_h=4.0, center=(0, 0.35, 0))
        names = ["Input"] + ["Hidden"] * (len(sizes) - 2) + ["Output"]
        tags = []
        for layer, name in zip(net.input_layers, names):
            tag = label(name, 22, theme.MUTED)
            tag.move_to((layer.get_center()[0], net.get_bottom()[1] - 0.4, 0))
            tags.append(tag)
        ctx.play(Write(head), run_time=0.7)
        ctx.play(Create(net), run_time=min(2.5, max(1.2, ctx.remaining * 0.3)))
        ctx.play(*[FadeIn(t, shift=DOWN * 0.1) for t in tags], run_time=0.6)
        if forward_pass:
            ctx.play(net.make_forward_pass_animation(), run_time=max(1.5, min(6.0, ctx.remaining - 0.5)))
        ctx.wait(ctx.remaining)
