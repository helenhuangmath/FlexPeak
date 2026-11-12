"""Differential calling: normalisation, overdispersion from replicates, and
recovery of simulated gains and losses with the right direction."""

import numpy as np
import pytest

from flexpeak.diff import DiffParams, differential_from_coverage, estimate_rho, normalise
from flexpeak.diff.core import _bb_logpmf, _bb_sf_two_sided
from flexpeak.peaks import interval_overlap_bp, interval_total_bp
from flexpeak.simulate import simulate_differential

SIZES = {"chr1": 2_000_000, "chr2": 1_000_000}


def _f1(calls, truth):
    inter = interval_overlap_bp(calls, truth)
    p = inter / max(interval_total_bp(calls), 1)
    r = inter / max(interval_total_bp(truth), 1)
    return 2 * p * r / max(p + r, 1e-9), p, r


@pytest.fixture(scope="module")
def sim():
    return simulate_differential(chrom_sizes=SIZES, n_shared_narrow=20, n_shared_broad=2,
                                 n_diff_narrow=16, n_diff_broad=4, depth_b=0.6, seed=11)


def test_betabinomial_reduces_to_binomial_and_is_two_sided():
    from scipy import stats

    a, n = np.array([3.0, 10.0, 0.0]), np.array([10.0, 10.0, 0.0])
    bb = _bb_logpmf(a, n, 0.3, 0.0)
    bi = stats.binom.logpmf(a[:2], n[:2], 0.3)
    assert np.allclose(bb[:2], bi)
    assert bb[2] == 0.0                     # n == 0 is uninformative
    assert _bb_sf_two_sided(50, 100, 0.5, 0.0) == pytest.approx(1.0, abs=0.2)
    assert _bb_sf_two_sided(90, 100, 0.5, 0.0) < 1e-10
    # Overdispersion makes the same split less surprising.
    assert _bb_sf_two_sided(70, 100, 0.5, 0.05) > _bb_sf_two_sided(70, 100, 0.5, 0.0)


def test_normalisation_undoes_depth(sim):
    for method in ("total", "background", "tmm"):
        f = normalise(sim.reps_a + sim.reps_b, method)
        assert len(f) == 4
        # B was simulated at 60% depth: its factors must be larger than A's.
        assert min(f[2:]) > max(f[:2])
        assert 1.3 < np.mean(f[2:]) / np.mean(f[:2]) < 2.2


def test_rho_from_replicates_is_positive_and_small(sim):
    arrs = [r.concat() for r in sim.reps_a]
    rho = estimate_rho(arrs)
    assert rho is not None and 0.0 < rho < 0.2
    # Two identical copies have zero excess scatter.
    assert estimate_rho([arrs[0], arrs[0].copy()]) < 1e-3 or estimate_rho([arrs[0], arrs[0].copy()]) is None


@pytest.mark.parametrize("norm", ["background", "tmm"])
def test_recovers_gains_and_losses_with_direction(sim, norm):
    reps_a = [r.rebin(100) for r in sim.reps_a]
    reps_b = [r.rebin(100) for r in sim.reps_b]
    res = differential_from_coverage(reps_a, reps_b, params=DiffParams(norm=norm))
    assert len(res) > 0
    f1_g, p_g, r_g = _f1(res.intervals("gain"), sim.truth_intervals("gain"))
    f1_l, p_l, r_l = _f1(res.intervals("loss"), sim.truth_intervals("loss"))
    assert r_g > 0.8 and r_l > 0.8, (r_g, r_l)
    assert p_g > 0.7 and p_l > 0.7, (p_g, p_l)
    # Shared (non-differential) peaks must not be called.
    shared = [(r.chrom, r.start, r.end) for r in sim.shared]
    leak = interval_overlap_bp(res.intervals(), shared) / interval_total_bp(shared)
    assert leak < 0.1
    assert res.diagnostics["rho_source"] == "replicates"
    assert res.diagnostics["region_rho_source"].startswith("replicate scatter")
    assert 0 < res.diagnostics["region_rho"] < 0.3
    assert "differential report" in res.report()


def test_single_sample_per_condition_still_works(sim):
    res = differential_from_coverage([sim.reps_a[0].rebin(100)], [sim.reps_b[0].rebin(100)],
                                     params=DiffParams())
    assert res.diagnostics["rho_source"].startswith("EM")
    f1, p, r = _f1(res.intervals(), sim.truth_intervals())
    assert r > 0.7 and p > 0.5


def test_output_files_roundtrip(sim, tmp_path):
    res = differential_from_coverage([r.rebin(100) for r in sim.reps_a],
                                     [r.rebin(100) for r in sim.reps_b])
    paths = res.write_all(str(tmp_path / "d"))
    for k, p in paths.items():
        assert (tmp_path / p.split("/")[-1]).exists()
    lines = [l for l in open(paths["narrowPeak"]) if not l.startswith("#")]
    assert len(lines) == len(res)
    strands = {l.split("\t")[5] for l in lines}
    assert strands <= {"+", "-"}
    gain = sum(1 for l in open(paths["gain"]))
    assert gain == res.n_gain


def test_null_comparison_calls_nothing():
    s = simulate_differential(chrom_sizes=SIZES, n_diff_narrow=0, n_diff_broad=0,
                              n_shared_narrow=20, n_shared_broad=2, seed=5)
    res = differential_from_coverage([r.rebin(100) for r in s.reps_a],
                                     [r.rebin(100) for r in s.reps_b])
    assert len(res) <= 2


def test_no_replicate_rho_floor_and_pseudo_depth(sim):
    from flexpeak.signal.coverage import Coverage

    a = sim.reps_a[0].rebin(100)
    b = sim.reps_b[0].rebin(100)
    # Pretend the input was a normalised bigWig at an arbitrary scale.
    big = lambda c: Coverage(bins={k: v * 37.0 for k, v in c.bins.items()}, bin_size=100,
                             chrom_sizes=dict(c.chrom_sizes), source=c.source, is_count=False)
    res = differential_from_coverage([big(a)], [big(b)], params=DiffParams())
    assert res.diagnostics["rho"] >= 0.02
    assert "floored" in res.diagnostics["rho_source"]
    assert res.diagnostics["region_rho_source"].startswith("no replicates")
    assert res.diagnostics["is_count"] is False
    # 37x-scaled counts: the units estimate must undo the 37.  NB
    # overdispersion and the simulator's 35% baseline drift inflate it (to
    # ~4x here), never deflate it, so the recovered counts err small and the
    # test stays conservative.
    from flexpeak.diff.core import _poisson_units
    u = _poisson_units(big(a))
    assert 37.0 <= u <= 37.0 * 8.0, u
    f1, p, r = _f1(res.intervals(), sim.truth_intervals())
    assert r > 0.6 and p > 0.5
    shared = [(x.chrom, x.start, x.end) for x in sim.shared]
    assert interval_overlap_bp(res.intervals(), shared) / interval_total_bp(shared) < 0.15


def test_depth_adaptive_rebin_on_sparse_input(sim):
    from flexpeak.signal.coverage import Coverage

    # 1/8 of the simulated depth: ~4 fragments per 100 bp bin per condition.
    thin = lambda c: Coverage(bins={k: (v / 8.0).astype(np.float32) for k, v in c.bins.items()},
                              bin_size=100, chrom_sizes=dict(c.chrom_sizes), source=c.source)
    ra = [thin(r.rebin(100)) for r in sim.reps_a]
    rb = [thin(r.rebin(100)) for r in sim.reps_b]
    res = differential_from_coverage(ra, rb, params=DiffParams())
    assert res.diagnostics["rebin_note"] is not None
    assert res.diagnostics["bin_size"] >= 200
    f1, p, r = _f1(res.intervals(), sim.truth_intervals())
    assert r > 0.7 and p > 0.6, (p, r)
    off = differential_from_coverage(ra, rb, params=DiffParams(min_depth_per_bin=0))
    assert off.diagnostics["rebin_note"] is None and off.diagnostics["bin_size"] == 100


def test_enrichment_gate_drops_background_candidates(sim):
    ra = [r.rebin(100) for r in sim.reps_a]; rb = [r.rebin(100) for r in sim.reps_b]
    gated = differential_from_coverage(ra, rb, params=DiffParams())
    open_ = differential_from_coverage(ra, rb, params=DiffParams(require_enrichment=False))
    assert gated.diagnostics["enrichment_gate"] == "pooled local background"
    assert open_.diagnostics["enrichment_gate"] == "none"
    # Simulated differential regions are all enriched: the gate must keep them.
    f1, p, r = _f1(gated.intervals(), sim.truth_intervals())
    assert r > 0.95 and p > 0.85, (p, r)
    assert len(gated) <= len(open_)
