import numpy as np
from flexpeak.diff import DiffParams
from flexpeak.diff.core import normalise
from flexpeak.background.nb import local_lambda
from flexpeak.simulate import simulate_differential
SIZES = {"chr1": 2_000_000, "chr2": 1_000_000}
sim = simulate_differential(chrom_sizes=SIZES, n_shared_narrow=20, n_shared_broad=2, n_diff_narrow=16, n_diff_broad=4, depth_b=0.6, seed=11)
ra = [r.rebin(100) for r in sim.reps_a]; rb = [r.rebin(100) for r in sim.reps_b]
f = normalise(ra + rb, "background"); fa, fb = f[:2], f[2:]
c = "chr1"
a = np.mean([r.bins[c].astype(float) * x for r, x in zip(ra, fa)], axis=0)
b = np.mean([r.bins[c].astype(float) * x for r, x in zip(rb, fb)], axis=0)
pooled = 0.5 * (a + b)
print("pooled mean", pooled.mean(), "median", np.median(pooled))
lam = local_lambda(pooled, 100, (10_000, 50_000), float(pooled.mean()), exclude_fold=2.0)
print("lambda: mean", lam.mean(), "min", lam.min(), "max", lam.max(), "p50", np.median(lam), "p99", np.percentile(lam, 99))
for t in sim.truth[:10]:
    if t.chrom != c: continue
    s, e = t.start // 100, t.end // 100
    ma, mb, ls = a[s:e].sum(), b[s:e].sum(), lam[s:e].sum()
    print(f"{t.direction:4s} {t.kind:6s} {t.start:>8}-{t.end:<8} ma {ma:8.0f} mb {mb:8.0f} lam_sum {ls:8.0f} ratio_a {ma/ls:5.2f} ratio_b {mb/ls:5.2f}")
