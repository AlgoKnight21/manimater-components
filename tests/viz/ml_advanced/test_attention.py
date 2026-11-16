import numpy as np
import pytest

from manimate.viz.core.palette import PALETTE
from manimate.viz.core.plumbing import dry_run
from manimate.viz.ml_advanced.attention import AttentionArcs, AttentionHeatmap, TokenStrip

RNG = np.random.default_rng(3)


def run(anim):
    dry_run(lambda ctx: ctx.play(anim, run_time=1.0))


def test_token_strip_has_one_box_per_token_and_optional_embedding_bars():
    s = TokenStrip(["The", "cat", "sat"], embeddings=RNG.normal(size=(3, 4)), highlight=1)
    assert len(s.boxes) == 3 and len(s.bars) == 3 and len(s.bars[0]) == 4
    with pytest.raises(ValueError):
        TokenStrip(["a"] * 13)
    with pytest.raises(ValueError, match="one row per token"):
        TokenStrip(["a", "b"], embeddings=np.zeros((3, 2)))
    run(s.animate_highlight(2))


def test_heatmap_walks_raw_scaled_masked_softmax():
    Q, K = RNG.normal(size=(4, 4)), RNG.normal(size=(4, 4))
    h = AttentionHeatmap(Q, K, ["a", "b", "c", "d"], causal=True)
    assert h.stage == "raw" and h.numbers is not None
    run(h.animate_scale()); run(h.animate_mask())
    assert h.stage == "masked" and h.grid.cells[0][3].get_fill_color().to_hex().lower() == PALETTE.bg.lower()
    assert h.grid.cells[3][0].get_fill_color().to_hex().lower() != PALETTE.bg.lower()
    run(h.animate_softmax())
    assert h.stage == "softmax"
    assert np.allclose(h.data.weights.sum(axis=1), 1.0) and np.all(np.triu(h.data.weights, 1) == 0)


def test_heatmap_without_causal_skips_the_mask_and_checks_cell_indices():
    Q = K = RNG.normal(size=(3, 2))
    h = AttentionHeatmap(Q, K)
    assert type(h.animate_mask()).__name__ == "Wait"
    run(h.animate_dot_product(1, 2)); run(h.animate_clear())
    with pytest.raises(ValueError):
        h.animate_dot_product(5, 0)
    with pytest.raises(ValueError, match="at most"):
        AttentionHeatmap(RNG.normal(size=(17, 2)), RNG.normal(size=(17, 2)))


def test_arcs_get_thicker_with_weight_and_top_k_limits_them():
    a = AttentionArcs(["a", "b", "c", "d"], [0.1, 0.2, 0.3, 0.4], query_index=3)
    widths = [arc.get_stroke_width() for arc in a.arcs]
    assert len(a.arcs) == 3 and widths == sorted(widths)
    assert len(AttentionArcs(["a", "b", "c", "d"], [0.1, 0.2, 0.3, 0.4], 3, top_k=1).arcs) == 1
    with pytest.raises(ValueError):
        AttentionArcs(["a", "b"], [0.5, 0.5, 0.0], 0)
    run(a.animate_in())
