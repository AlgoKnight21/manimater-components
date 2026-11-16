"""T0: pure-maths invariants for viz.microelectronics (README section 14.1)."""
import numpy as np
import pytest

from manimate.viz.microelectronics import _math as m


def test_depletion_width_grows_from_forward_to_reverse_and_is_one_at_zero_bias():
    vs = np.linspace(0.6, -2.0, 30)
    ws = [m.depletion_width(v)[0] for v in vs]
    assert all(b >= a for a, b in zip(ws, ws[1:]))
    assert m.depletion_width(0.0)[0] == pytest.approx(1.0)
    xp, xn = m.depletion_split(1.0, na=4.0, nd=1.0)
    assert xp + xn == pytest.approx(1.0) and xp < xn  # the lightly doped side depletes further


def test_diode_current_is_tiny_in_reverse_and_exponential_in_forward():
    assert abs(m.diode_current(-1.0)) < 1.1e-12
    assert m.diode_current(0.7) > 1e3 * m.diode_current(0.5)
    assert np.isfinite(m.diode_current(50.0))


def test_mos_region_boundaries_and_current_continuity():
    assert m.mos_region(0.5, 1.0, 1.0) == "cutoff"
    assert m.mos_region(2.0, 0.99, 1.0) == "triode"
    assert m.mos_region(2.0, 1.0, 1.0) == "saturation"
    lo, hi = m.drain_current(2.0, 1.0 - 1e-9, 1.0, 0.5), m.drain_current(2.0, 1.0, 1.0, 0.5)
    assert lo == pytest.approx(hi, rel=1e-6)
    assert m.drain_current(0.5, 2.0, 1.0) == 0.0


def test_channel_profile_pinches_off_in_saturation():
    xs, depth = m.channel_profile(2.0, 0.0, 1.0)
    assert np.allclose(depth, depth[0])
    xs, depth = m.channel_profile(2.0, 3.0, 1.0)  # saturation: the layer vanishes before the drain
    assert depth[-1] == 0.0 and depth[0] > 0
    assert m.pinch_point(2.0, 3.0, 1.0) == pytest.approx(1 / 3)
    assert m.pinch_point(0.5, 1.0, 1.0) is None


def test_independent_program_has_no_stalls_and_takes_n_plus_four_cycles():
    s = m.pipeline_schedule(m.resolve_program("independent"), forwarding=False)
    assert s.stall_cycles == 0 and s.total_cycles == 4 + 4


def test_raw_hazard_without_forwarding_costs_two_bubbles_and_forwarding_removes_them():
    prog = m.resolve_program("raw_hazard")
    slow, fast = m.pipeline_schedule(prog, forwarding=False), m.pipeline_schedule(prog, forwarding=True)
    assert slow.stall_cycles == 2 and slow.total_cycles == 10
    assert fast.stall_cycles == 0 and fast.total_cycles == 8
    assert fast.forwards and all(src < dst for src, dst, *_ in fast.forwards)
    assert any(c.cause == "RAW" for c in slow.rows[1].cells if c.kind == "stall")


def test_load_use_still_costs_one_bubble_with_forwarding():
    s = m.pipeline_schedule(m.resolve_program("load_use"), forwarding=True)
    assert s.stall_cycles == 1
    assert [c.cause for c in s.rows[1].cells if c.kind == "stall"] == ["load-use"]


def test_branch_policies():
    prog = m.resolve_program("branch")
    stall = m.pipeline_schedule(prog, branch_policy="stall")
    pnt = m.pipeline_schedule(prog, branch_policy="predict_not_taken")  # the branch is taken: a mispredict
    ptk = m.pipeline_schedule(prog, branch_policy="predict_taken")
    control = sum(1 for r in stall.rows for c in r.cells if c.cause == "control")
    assert control == 2  # the branch resolves in EX: two wasted fetch slots (the rest of stall_cycles is the lw/add load-use)
    assert pnt.flush_cycles == 2 and sum(r.squashed for r in pnt.rows) == 2
    assert ptk.flush_cycles == 1


def test_predictors():
    d = m.two_bit_predictor([1, 1, 0, 1, 1])
    assert d.states == (1, 2, 3, 2, 3) and d.predictions == (0, 1, 1, 1, 1) and d.final_state == 3
    assert d.mispredicts == 2
    assert m.predictor_run([1, 1, 1], "static").mispredicts == 3
    assert m.predictor_run([1, 0, 1], "1bit").predictions == (0, 1, 0)
    with pytest.raises(ValueError):
        m.predictor_run([2])


def test_cache_matches_a_hand_trace():
    # 2 sets x 1 way, 4-byte blocks: addresses 0 and 8 both map to set 0 and evict each other; 4 maps to set 1.
    r = m.cache_simulate([0, 4, 8, 0, 4], sets=2, ways=1, block=4)
    assert [a.hit for a in r.accesses] == [False, False, False, False, True]
    assert (r.accesses[0].tag, r.accesses[0].index, r.accesses[0].offset) == (0, 0, 0)
    assert (r.accesses[2].tag, r.accesses[2].index, r.accesses[2].evicted_tag) == (1, 0, 0)
    assert r.hits == 1 and r.misses == 4


def test_lru_versus_fifo_differ_on_a_reuse_pattern():
    seq = [0, 8, 0, 16, 8]  # one set, two ways, block 8 -> tags 0, 1, 0, 2, 1
    lru = m.cache_simulate(seq, sets=1, ways=2, block=8, policy="lru")
    fifo = m.cache_simulate(seq, sets=1, ways=2, block=8, policy="fifo")
    assert lru.accesses[3].evicted_tag == 1 and fifo.accesses[3].evicted_tag == 0
    with pytest.raises(ValueError):
        m.cache_simulate([0], sets=3)


def test_amat_single_and_multi_level():
    assert m.amat(1, 0.1, 100) == pytest.approx(11.0)
    assert m.amat([1, 10], [0.1, 0.5], 100) == pytest.approx(1 + 0.1 * (10 + 0.5 * 100))


def test_dma_always_finishes_and_burst_is_contiguous():
    s = m.dma_schedule(8, cpu_busy=1.0)
    assert s.owners.count("DMA") == 8
    assert m.dma_schedule(5, mode="burst").owners == ("DMA",) * 5


def test_spi_sampling_instants_per_mode():
    for cpol in (0, 1):
        for cpha in (0, 1):
            tl = m.spi_frame(0xA5, cpol, cpha)
            sig = tl.signals["SCLK"]
            assert sig[0][1] == cpol  # idle level
            got = [b for _, b, name in tl.samples if name == "MOSI"]
            assert got == [1, 0, 1, 0, 0, 1, 0, 1]
            for t, b, _ in tl.samples:  # the data line is stable at every sampling instant
                assert m.level_at(tl.signals["MOSI"], t - 0.2) == b == m.level_at(tl.signals["MOSI"], t + 0.2) or cpha == 0
            # cpha 0 samples on the leading edge; cpha 1 on the trailing edge
            edges = [t for t, lv in sig if lv != cpol]  # leading edges move away from idle
            assert (tl.samples[0][0] in edges) == (cpha == 0)


def test_i2c_start_stop_and_ack():
    tl = m.i2c_frame(0x48, 0xA5, "w")
    sda, scl = tl.signals["SDA"], tl.signals["SCL"]
    assert m.level_at(scl, 0.6) == 1 and m.level_at(sda, 0.4) == 1 and m.level_at(sda, 0.6) == 0  # START
    assert [f[2] for f in tl.fields][:4] == ["START", "ADDR 0x48", "R/W", "ACK"]
    assert tl.fields[-1][2] == "STOP" and m.level_at(sda, tl.duration - 0.1) == 1
    assert [b for _, b, _ in tl.samples][:7] == [1, 0, 0, 1, 0, 0, 0]


def test_uart_frame_layout_and_parity():
    tl = m.uart_frame(0x41, parity="even", stop=1)  # 'A' = 0b01000001 has two ones -> even parity bit is 0
    names = [f[2] for f in tl.fields]
    assert names[:3] == ["IDLE", "START", "D0"] and "PAR" in names and names.count("STOP") == 1
    assert [b for _, b, _ in tl.samples][:2] == [0, 1]  # start bit then LSB of 0x41
    assert [b for _, b, _ in tl.samples if True][9 + 0] == 0  # parity bit
    assert m.uart_frame(0x01, parity="odd").samples[9][1] == 0
    assert m.uart_frame(0x00, parity="odd").samples[9][1] == 1


def test_usb_nrzi_and_stuffing():
    r = m.nrzi_encode([1] * 14)
    run = best = 0
    for b in r.data:
        run = run + 1 if b else 0
        best = max(best, run)
    assert best <= 6 and len(r.stuffed) == 2
    assert all(r.data[i] == 0 for i in r.stuffed)
    assert m.nrzi_encode([0, 1, 1], stuffing=False).levels == (0, 0, 0)  # 0 toggles, 1 holds
    assert m.nrzi_encode([1, 1], stuffing=False).levels == (1, 1)


def test_can_stuffing_never_leaves_six_equal_bits_and_crc_is_stable():
    for can_id in (0x000, 0x7FF, 0x123, 0x555):
        f = m.can_frame(can_id, [0x00, 0xFF, 0x00])
        end = next(a for a, _, n in f.fields if n == "CRC del")
        run, last, worst = 0, None, 0
        for b in f.bits[:end]:
            run = run + 1 if b == last else 1
            last = b
            worst = max(worst, run)
        assert worst <= 5, (can_id, worst)
    assert m.can_frame(0x123, [1, 2]).crc == m.can_frame(0x123, [1, 2]).crc
    assert m.can_frame(0x123, stuffing=False).bits != m.can_frame(0x000, stuffing=False).bits


def test_can_arbitration_lowest_id_wins():
    r = m.can_arbitration([0x120, 0x100, 0x180])
    assert r.winner == 1 and r.lost_at[1] is None
    assert r.lost_at[0] is not None and r.lost_at[2] is not None
    assert min(r.ids) == r.ids[r.winner]
    with pytest.raises(ValueError):
        m.can_arbitration([5, 5])


def test_differential_cancels_common_mode_noise():
    d = m.differential([1, 0, 1, 1, 0], noise=0.4)
    clean = m.differential([1, 0, 1, 1, 0], noise=0.0)
    assert np.allclose(d.diff, clean.diff)
    assert np.std(d.common) > 0.05
    assert np.allclose(d.dplus - clean.dplus, d.dminus - clean.dminus)  # the same noise lands on both lines
