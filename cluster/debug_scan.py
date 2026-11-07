import numpy as np
from scipy.ndimage import uniform_filter1d
from flexpeak.scan import profile_bins, sigma_clip, _drift_corrected, scan_coverage
from flexpeak.simulate import simulate
rng = np.random.default_rng(1)
x = rng.negative_binomial(4, 4 / (4 + 6.0), size=100_000).astype(float)
for s_bp in (1000, 5000):
    k = s_bp // 25
    sm = uniform_filter1d(x, size=k, mode="nearest")
    dc, _ = _drift_corrected(x, sm, 25, s_bp)
    for name, arr in (("raw sm", sm), ("drift-corr", dc)):
        m, s, keep = sigma_clip(arr, 5.0, averaged_over=k)
        print(f"null s={s_bp} {name:10s} mean {arr.mean():.3f} sd {arr.std():.3f} | clip mean {m:.3f} sd {s:.3f} kept {keep.mean():.4f} tail {(arr > m + 5*s).mean():.5f}")
p = profile_bins(x, 25, n_sd=5.0); print("null profile excess", p.signal_excess, p.smoothed_excess)
SIZES = {"chr1": 2_000_000, "chr2": 500_000}
sb = simulate(chrom_sizes=SIZES, seed=7, dispersion=0.15, drift_sd=0.15, n_narrow=0, n_broad=6, n_mixed=0, broad_fold=2.5, broad_width=80_000)
xb = sb.treatment.bins["chr1"].astype(float)
truth = np.zeros(xb.size, bool)
for t in sb.truth:
    if t.chrom == "chr1": truth[t.start//25:t.end//25] = True
print("broad: domain fraction", truth.mean())
for s_bp in (1000, 5000):
    k = s_bp // 25
    sm = uniform_filter1d(xb, size=k, mode="nearest")
    dc, _ = _drift_corrected(xb, sm, 25, s_bp)
    for name, arr in (("raw sm", sm), ("drift-corr", dc)):
        m, s, keep = sigma_clip(arr, 5.0, averaged_over=k)
        above = arr > m + 5*s
        print(f"broad s={s_bp} {name:10s} bg mean {arr[~truth].mean():.2f} sd {arr[~truth].std():.2f} dom mean {arr[truth].mean():.2f} | clip mean {m:.2f} sd {s:.2f} thr {m+5*s:.2f} tail {above.mean():.4f} (in domains {above[truth].mean():.3f})")
print(scan_coverage(sb.treatment, (25, 50, 100), chrom="chr1").report())

sn = simulate(chrom_sizes=SIZES, seed=7, dispersion=0.15, drift_sd=0.15, n_narrow=0, n_broad=0, n_mixed=0)
print("NULL drift0.15:", scan_coverage(sn.treatment, (25, 50, 100), chrom="chr1").recommend().signal_class)
sb35 = simulate(chrom_sizes=SIZES, seed=7, dispersion=0.15, n_narrow=0, n_broad=6, n_mixed=0, broad_fold=2.5, broad_width=80_000)
pr = scan_coverage(sb35.treatment, (25, 50, 100), chrom="chr1"); print("BROAD drift0.35:", pr.recommend().signal_class, pr.profiles[100].smoothed_excess)
for kind, kw in (("narrow", dict(n_narrow=60, n_broad=0, n_mixed=0, narrow_fold=10.0)),
                 ("mixed", dict(n_narrow=20, n_broad=2, n_mixed=2))):
    sx = simulate(chrom_sizes=SIZES, seed=7, dispersion=0.15, drift_sd=0.15, **kw)
    pr = scan_coverage(sx.treatment, (25, 50, 100), chrom="chr1"); r = pr.recommend()
    print(f"{kind.upper()}: class {r.signal_class} bin {r.bin_size} windows {r.windows} seg {r.segment_bin_size} scale {r.scale_bp} max_gap {r.max_gap}")
    for line in r.reasons: print("    -", line)
import json, glob
from flexpeak.scan import ScanProfile, BinProfile
for f in sorted(glob.glob("testing/results/real/scan_*.json")):
    d = json.load(open(f))
    profs = {int(b): BinProfile(**{k: ({int(x): y for x, y in v.items()} if isinstance(v, dict) else v) for k, v in p.items()}) for b, p in d["profiles"].items()}
    sp = ScanProfile(chrom=d["chrom"], chrom_size=d["chrom_size"], source=d["source"], n_sd=d["n_sd"], is_count=d["is_count"], profiles=profs)
    r = sp.recommend()
    print(f"REAL {f.split('scan_')[1][:-5]:<14} class {r.signal_class:<7} bin {r.bin_size:>3} windows {r.windows} seg {r.segment_bin_size} scale {r.scale_bp} max_gap {r.max_gap}")
