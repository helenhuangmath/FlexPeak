"""Stratified multi-chromosome scan, robust background start, weak-domain
pass, replicate-group tuning halves and the coordinate search."""

import numpy as np

from flexpeak import CallParams, call_peaks
from flexpeak.background.nb import local_lambda
from flexpeak.caller import _weak_domains, weak_domain_window_bp
from flexpeak.scan import plan_scan_windows, scan_coverage
from flexpeak.signal.coverage import Coverage
from flexpeak.simulate import simulate
from flexpeak.tune.sweep import pick_tuning_chroms, replicate_halves, tune

MM = {"chr1": 195_471_971, "chr2": 182_113_224, "chr4": 156_508_116,
      "chr7": 145_441_459, "chr10": 130_694_993, "chr11": 122_082_543,
      "chr15": 104_043_685, "chr19": 61_431_566, "chrX": 171_031_299,
      "chrY": 91_744_698, "chrUn_GL456239": 40_056, "chr1_random": 100_000}


def test_plan_scan_windows_is_stratified_and_keeps_the_budget():
    plan = plan_scan_windows(MM, 60_000_000, 6)
    chroms = [c for c, _, _ in plan]
    assert len(plan) == 6
    assert "chr1" in chroms and "chr19" in chroms           # largest and smallest
    assert not any(c in ("chrY", "chrUn_GL456239", "chr1_random") for c in chroms)
    assert sum(e - s for _, s, e in plan) == 60_000_000
    for c, s, e in plan:
        assert 0 < s < e <= MM[c]
    # Fewer chromosomes than windows: each is taken whole when small enough.
    small = plan_scan_windows({"chr1": 2_000_000, "chr2": 500_000}, 60_000_000, 6)
    assert small == [("chr1", 0, 2_000_000), ("chr2", 0, 500_000)]


def test_multi_window_scan_profiles_every_window():
    sim = simulate(chrom_sizes={"chr1": 2_000_000, "chr2": 1_000_000},
                   n_narrow=40, n_broad=0, n_mixed=0, seed=5)
    prof = scan_coverage(sim.treatment, (25, 50, 100))
    assert [r[0] for r in prof.regions] == ["chr1", "chr2"]
    assert prof.profiled_bp == 3_000_000
    assert prof.profiles[25].n_bins == 3_000_000 // 25
    assert "windows" in prof.report()


def test_robust_start_breaks_the_moderate_enrichment_fixed_point():
    # A domain mark's chromosome: background 2.4, 30% of it in 20 kb
    # stretches at 3x, a 350 kb domain at 26 and a 23 kb block at 9.7 next
    # to it (GSE285245 H3K27me3 chr11 levels).  The plain iteration starts at
    # the mean (~3.8), keeps everything below twice that -- the 3x stretches
    # included -- and stays there.
    rng = np.random.default_rng(0)
    n = 120_000
    lam = np.full(n, 2.4)
    starts = rng.choice(np.arange(0, n - 200, 200), size=int(0.30 * n / 200), replace=False)
    for st in starts:
        lam[st:st + 200] = 7.2
    lam[60_000:63_500] = 26.0
    lam[59_270:59_500] = 9.7
    lam[59_500:60_000] = 2.4
    x = rng.poisson(lam).astype(np.float32)
    W = (1_000_000, 5_000_000)
    plain = local_lambda(x, 100, W, float(x.mean()))
    robust = local_lambda(x, 100, W, float(x.mean()), robust_start=True)
    blk = slice(59_270, 59_500)
    # The plain estimate is the mean of background plus the stretches; the
    # anchored one stays at the background.
    assert float(plain[blk].mean()) > 1.25 * 2.4
    assert float(robust[blk].mean()) < 1.1 * 2.4
    assert float(x[blk].mean() / robust[blk].mean()) > 3.5
    # On background alone the anchor changes nothing.
    y = rng.poisson(np.full(n, 5.0)).astype(np.float32)
    a = local_lambda(y, 100, W, float(y.mean()))
    b = local_lambda(y, 100, W, float(y.mean()), robust_start=True)
    assert abs(float(b.mean()) / float(a.mean()) - 1.0) < 0.02


def _bg(a0=0.05):
    from flexpeak.background.nb import BackgroundModel

    return BackgroundModel(mu={}, a0=a0, a1=0.0, genome_mean=5.0, windows=(),
                           used_control=False, autocorr_inflation=1.5)


def test_weak_domain_pass_finds_long_low_patchy_enrichment():
    rng = np.random.default_rng(1)
    n = 200_000
    mu = np.full(n, 5.0)
    lam = mu.copy()
    seg = np.arange(80_000, 80_650)                   # 65 kb at 2x, with holes
    lam[seg] = 10.0
    lam[seg[::7]] = 2.0
    counts = rng.poisson(lam).astype(np.float64)
    p = CallParams(windows=(1_000_000, 5_000_000), domain_min_width=3000)
    assert weak_domain_window_bp(p) == 10_000
    out = _weak_domains(counts, mu, _bg(), 100, 10_000, p, n * 100)
    assert len(out) == 1
    s, e = out[0]
    assert abs(s - 80_000) < 80 and abs(e - 80_650) < 80
    # Background steps are not domains: a quiet stretch at 2 inside a
    # background at 5, the Pdcd1 H3K9me3 geometry.
    step = np.full(n, 5.0)
    step[70_000:90_000] = 2.0
    stepc = rng.poisson(step).astype(np.float64)
    assert _weak_domains(stepc, np.full(n, 5.0), _bg(), 100, 10_000, p, n * 100) == []
    null = rng.poisson(mu).astype(np.float64)
    assert _weak_domains(null, mu, _bg(), 100, 10_000, p, n * 100) == []
    assert weak_domain_window_bp(CallParams()) == 0     # narrow-scale windows: off


def test_replicate_halves_and_coordinate_search():
    sizes = {"chr1": 1_500_000, "chr2": 800_000}
    pooled = simulate(chrom_sizes=sizes, n_narrow=25, n_broad=2, n_mixed=1, seed=11).treatment
    h0, h1 = pooled.split(seed=0)
    reps = [*h0.split(seed=1), *h1.split(seed=2)]
    total = sum(float(a.sum()) for a in pooled.bins.values())
    a, b, desc = replicate_halves(reps, list(sizes), pooled.bin_size, total)
    assert desc == "replicates 1,3 | 2,4"
    for h in (a, b):
        assert abs(sum(float(v.sum()) for v in h.bins.values()) - 0.5 * total) < 1e-3 * total
    grid = {"min_fold": (1.2, 1.5), "posterior_cutoff": (0.4, 0.5),
            "max_gap": (0, 500), "weak_domain_bp": (0, 10_000)}
    res = tune(pooled, base=CallParams(bin_size=50), grid=grid, chroms=list(sizes),
               replicates=reps)
    assert res.notes[0] == "halves: replicates 1,3 | 2,4"
    assert any(n.startswith("coordinate search") for n in res.notes)
    # Replicate support is the reproducibility term: simulated peaks are real.
    assert res.score.reproducibility > 0.5
    assert all(sc.detail["has_replicates"] for _, sc in res.curve)
    assert len(res.curve) < 16                          # fewer than the full product
    for k in grid:
        assert getattr(res.params, k) in grid[k]
    call_peaks(pooled, None, res.params)


def test_tuning_chroms_skip_largest_and_sex_chromosomes():
    cov = Coverage(bins={c: np.zeros(100, np.float32) for c in MM}, bin_size=10,
                   chrom_sizes=dict(MM))
    picks = pick_tuning_chroms(cov, 2, skip_largest=True)
    assert "chr1" not in picks and "chrX" not in picks and "chrY" not in picks
    assert "chr19" in picks and len(picks) == 2


def test_tune_on_stratified_windows():
    from flexpeak.tune.sweep import tuning_windows, window_coverage

    sizes = {"chr1": 1_500_000, "chr2": 1_200_000, "chr3": 900_000, "chr4": 600_000}
    pooled = simulate(chrom_sizes=sizes, n_narrow=30, n_broad=2, n_mixed=1, seed=4).treatment
    plan = tuning_windows(pooled, total_bp=2_000_000, n=4)
    assert len(plan) == 4 and sum(e - s for _, s, e in plan) <= 2_000_000
    wc = window_coverage(pooled, plan)
    assert list(wc.bins) == [f"{c}:{s}-{e}" for c, s, e in plan]
    h0, h1 = pooled.split(seed=0)
    reps = [*h0.split(seed=1), *h1.split(seed=2)]
    res = tune(pooled, base=CallParams(bin_size=50), windows=plan, replicates=reps,
               grid={"min_fold": (1.2, 1.5), "posterior_cutoff": (0.4, 0.5)})
    assert res.chroms == list(wc.bins)
    assert res.notes[0] == "halves: replicates 1,3 | 2,4"
