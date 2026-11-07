import numpy as np
from flexpeak.diff import DiffParams, differential_from_coverage
from flexpeak.diff.core import _fit_hmm, _decode, estimate_rho, normalise, _rep_arrays
from flexpeak.simulate import simulate_differential
from flexpeak.peaks import interval_overlap_bp, interval_total_bp
SIZES = {"chr1": 2_000_000, "chr2": 1_000_000}
sim = simulate_differential(chrom_sizes=SIZES, n_shared_narrow=20, n_shared_broad=2,
                            n_diff_narrow=16, n_diff_broad=4, depth_b=0.6, seed=11)
ra = [r.rebin(100) for r in sim.reps_a]; rb = [r.rebin(100) for r in sim.reps_b]
for norm in ("background", "tmm"):
    res = differential_from_coverage(ra, rb, params=DiffParams(norm=norm), log=print)
    d = res.diagnostics
    print(norm, {k: d[k] for k in ("rho","p_gain","p_loss","dwell_floor_bp","fitted_dwell_bp","n_candidates","scale_bp","n_iter")})
    for direction in ("gain", "loss"):
        calls, truth = res.intervals(direction), sim.truth_intervals(direction)
        inter = interval_overlap_bp(calls, truth)
        print(f"  {direction}: n={len(calls)} prec={inter/max(interval_total_bp(calls),1):.3f} rec={inter/max(interval_total_bp(truth),1):.3f}")
# posterior at truth regions
fa = normalise(ra + rb, "background")
a = np.mean([r.bins["chr1"]*f for r, f in zip(ra, fa[:2])], axis=0)
b = np.mean([r.bins["chr1"]*f for r, f in zip(rb, fa[2:])], axis=0)
n = np.rint(a+b); ai = np.minimum(np.rint(a), n)
for t in sim.truth[:8]:
    if t.chrom != "chr1": continue
    s, e = t.start//100, t.end//100
    print(f"truth {t.direction:4s} {t.kind:6s} {t.start}-{t.end} meanA {a[s:e].mean():.1f} meanB {b[s:e].mean():.1f} split {a[s:e].sum()/(a[s:e].sum()+b[s:e].sum()):.2f}")
print("rho within A:", estimate_rho([r.concat() for r in ra]), "within B:", estimate_rho([r.concat() for r in rb]),
      "cross A1/B1:", estimate_rho([ra[0].concat(), rb[0].concat()]))
from flexpeak.scan import scan_coverage
from flexpeak.simulate import simulate
sb = simulate(chrom_sizes=SIZES, seed=7, dispersion=0.15, n_narrow=0, n_broad=6, n_mixed=0, broad_fold=2.5, broad_width=80_000)
prof = scan_coverage(sb.treatment, (25, 50, 100), chrom="chr1"); print(prof.report())
