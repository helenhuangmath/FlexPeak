import numpy as np
from flexpeak.diff import DiffParams, differential_from_coverage
from flexpeak.simulate import simulate_differential
from flexpeak.signal.coverage import Coverage
from flexpeak.peaks import interval_overlap_bp, interval_total_bp
SIZES = {"chr1": 2_000_000, "chr2": 1_000_000}
sim = simulate_differential(chrom_sizes=SIZES, n_shared_narrow=20, n_shared_broad=2, n_diff_narrow=16, n_diff_broad=4, depth_b=0.6, seed=11)
ra = [r.rebin(100) for r in sim.reps_a]; rb = [r.rebin(100) for r in sim.reps_b]
big = lambda c: Coverage(bins={k: v * 37.0 for k, v in c.bins.items()}, bin_size=100, chrom_sizes=dict(c.chrom_sizes), source=c.source, is_count=False)
shared = [(x.chrom, x.start, x.end) for x in sim.shared]
def ev(res, tag):
    for d in ("gain", "loss", None):
        calls, truth = res.intervals(d), sim.truth_intervals(d)
        inter = interval_overlap_bp(calls, truth)
        p = inter / max(interval_total_bp(calls), 1); r = inter / max(interval_total_bp(truth), 1)
        print(f"  {tag:<34} {str(d):<5} n={len(calls):>3} prec={p:.3f} rec={r:.3f}" + (f"  leak={interval_overlap_bp(calls, shared)/interval_total_bp(shared):.3f}" if d is None else ""))
    d = res.diagnostics
    print(f"     rho bin {d['rho']:.4f} ({d['rho_source'][:20]}) region {d['region_rho']:.4f} ({d['region_rho_source'][:30]}) cands {d['n_candidates']}")
for scope in ("genome", "candidates"):
    print(f"=== fdr_scope={scope}")
    ev(differential_from_coverage(ra, rb, params=DiffParams(norm="background", fdr_scope=scope)), "2 reps, counts")
    ev(differential_from_coverage([ra[0]], [rb[0]], params=DiffParams(fdr_scope=scope)), "1 rep, counts")
    ev(differential_from_coverage([big(ra[0])], [big(rb[0])], params=DiffParams(fdr_scope=scope)), "1 rep, bigWig-like")
    ev(differential_from_coverage([big(r) for r in ra], [big(r) for r in rb], params=DiffParams(fdr_scope=scope)), "2 reps, bigWig-like")
# null with replicates
s0 = simulate_differential(chrom_sizes=SIZES, n_diff_narrow=0, n_diff_broad=0, n_shared_narrow=20, n_shared_broad=2, seed=5)
for scope in ("genome", "candidates"):
    res = differential_from_coverage([r.rebin(100) for r in s0.reps_a], [r.rebin(100) for r in s0.reps_b], params=DiffParams(fdr_scope=scope))
    print(f"NULL scope={scope}: {len(res)} regions, cands {res.diagnostics['n_candidates']}, region rho {res.diagnostics['region_rho']:.4f}")
