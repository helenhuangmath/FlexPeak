"""Per-position view of the multi-scale focal test at chosen loci.

Usage: python cluster/debug_focal_scales.py MARK CHROMS  (loci hard-coded below)
Prints, per scale, 1 kb rows: window sum, local expected, fold, -log10 p, q, inside-region flag,
plus the per-scale dispersion fits and region outcome at the locus.
"""
import json, os, sys
import numpy as np
from flexpeak import load_signal, CallParams, call_peaks
from flexpeak.scan import scan_file

R = os.environ["REPO"]; CACHE = f"{R}/testing/cache"; DATA = f"{R}/testing/data"
mark = sys.argv[1]; chroms = sys.argv[2].split(",")
LOCI = {"H3K9me3": [("chr1", 93_985_000, 94_115_000)],
        "H3K4me3": [("chr11", 46_429_000, 46_480_000)],
        "H3K27me3": [("chr11", 52_250_000, 52_310_000)]}[mark]
f = f"{DATA}/GSE285245_pool_{mark}_Cl13.bw"
rec = scan_file(f, cache_dir=CACHE).recommend()
params = CallParams().replace(**rec.overrides())
extra = json.loads(os.environ.get("OVERRIDES", "{}"))
for k, v in extra.items():
    extra[k] = tuple(v) if isinstance(v, list) else v
params = params.replace(**extra)
print("params", {k: getattr(params, k) for k in ("bin_size", "windows", "focal_scales", "min_fold",
                                                  "posterior_cutoff", "focal_min_fold")}, flush=True)
cov = load_signal(f, bin_size=params.bin_size, chroms=chroms, cache_dir=CACHE)
cache = {"focal_debug": LOCI}
res = call_peaks(cov, None, params, return_details=True, segmentation_cache=cache)
fd = res.diagnostics.get("focal", {})
print("focal diag", json.dumps({k: v for k, v in fd.items()}, default=str))
print("regions", len(res.peaks), "subpeaks", res.peaks.n_subpeaks)
for c, s, e in LOCI:
    print(f"\n== {c}:{s}-{e} called:")
    for r in res.peaks:
        if r.chrom == c and r.end > s and r.start < e:
            print(f"   {r.kind:<6} {r.source:<5} {r.start:,}-{r.end:,} fold {r.fold_enrichment:.2f} "
                  f"q {r.neg_log10_q:.2f} subs {[(sp.start, sp.end) for sp in r.subpeaks]}")
for d in cache.get("focal_debug_out", []):
    seg = d["seg"]; step = max(1, 1000 // seg)
    print(f"\n-- scale {d['scale']} bp (seg {seg}); rows every {step * seg} bp, max over the row")
    for i in range(0, d["nlp"].size, step):
        sl = slice(i, i + step)
        j = i + int(np.argmax(d["nlp"][sl]))
        pos = (d["start_bin"] + j) * seg
        print(f"   {pos:>12,} sum {d['c_w'][j]:9.1f} exp {d['mu_w'][j]:8.1f} fold {d['c_w'][j] / d['mu_w'][j]:5.2f} "
              f"nlp {d['nlp'][j]:6.2f} cut {d['p_cut']:5.2f} {'SIG' if d['sig'][j] else '   '} {'IN' if d['inside'][j] else '  '}")
