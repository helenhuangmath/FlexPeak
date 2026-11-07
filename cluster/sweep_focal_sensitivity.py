"""Sensitivity vs specificity of the focal pass, over three switches.

  focal_floor_fraction  1.0 / 0.5 / 0.25   (background floor = fraction x median)
  focal_dispersion      moments / robust   (window-sum noise from all bg windows vs lower half)
  focal_fdr             pooled / per_scale

For each setting: H3K9me3 Cl13 (chr1, chr11) region/sub-peak counts and the
calls at Pdcd1, plus the simulated null / narrow / mixed sets run the way
--quick runs them without a control (scan -> overrides), scored with the
benchmark's own metrics.
"""
import itertools
import os
import sys
import time

import numpy as np

R = os.environ["REPO"]
sys.path.insert(0, os.path.join(R, "Benchmark"))
from flexpeak import load_signal, CallParams, call_peaks  # noqa: E402
from flexpeak.scan import scan_coverage, scan_file  # noqa: E402
import run_benchmark as rb  # noqa: E402
from metrics import evaluate, evaluate_null  # noqa: E402

CACHE = f"{R}/testing/cache"
BW = f"{R}/testing/data/GSE285245_pool_H3K9me3_Cl13.bw"
PDCD1 = ("chr1", 93_938_000, 94_153_000)

GRID = list(itertools.product((1.0, 0.5, 0.25), ("moments", "clipped"), ("pooled",)))


def ivs(peaks):
    return [(r.chrom, r.start, r.end) for r in peaks]


def sub_ivs(peaks):
    out = []
    for r in peaks:
        out.extend((s.chrom, s.start, s.end) for s in r.subpeaks) if r.subpeaks else \
            out.append((r.chrom, r.start, r.end))
    return out


# -- simulated sets, treatment only, scan-derived parameters ----------------
specs = rb.dataset_specs(False)
sims = {}
for name in ("null", "narrow", "mixed"):
    d = rb.build_dataset(name, specs[name])
    prof = scan_coverage(d["treat"], (25, 50), chrom=sorted(d["treat"].bins)[0])
    rec = prof.recommend()
    base = CallParams().replace(**rec.overrides())
    sims[name] = (d, base, {})
    print(f"[sim {name}] class {rec.signal_class} windows {base.windows} focal {base.focal_scales} "
          f"bin {base.bin_size}", flush=True)

# -- real H3K9me3 ------------------------------------------------------------
rec = scan_file(BW, cache_dir=CACHE).recommend()
k9_base = CallParams().replace(**rec.overrides())
k9 = load_signal(BW, bin_size=k9_base.bin_size, chroms=["chr1", "chr11"], cache_dir=CACHE)
k9_cache = {}
print(f"[k9] windows {k9_base.windows} focal {k9_base.focal_scales}", flush=True)

# -- replicate truth proxy on chr1 (see check_k9_replicates.py) -------------
import glob  # noqa: E402
import pyBigWig  # noqa: E402
from scipy.ndimage import maximum_filter1d  # noqa: E402

REPBW = "/vast/projects/wherry/foundation-models-immuno/hhua/data_lab_internal/Huang_SR2025_CnR/geo_processed/per_sample_bw"


def _load(f, ch="chr1", bs=100):
    bw = pyBigWig.open(f)
    n = bw.chroms()[ch] // bs
    v = np.nan_to_num(np.array(bw.stats(ch, 0, n * bs, type="mean", nBins=n), dtype=float))
    bw.close()
    return v


def _ratio(x, k=2, flank=100, guard=2):
    c = np.concatenate([[0.0], np.cumsum(x)])
    n = x.size
    i = np.arange(n)
    lo, hi = np.clip(i - k // 2, 0, n), np.clip(i - k // 2 + k, 0, n)
    w = (c[hi] - c[lo]) / np.maximum(hi - lo, 1)
    elo, ehi = np.clip(lo - guard, 0, n), np.clip(hi + guard, 0, n)
    olo, ohi = np.clip(elo - flank, 0, n), np.clip(ehi + flank, 0, n)
    fl = (c[ohi] - c[olo] - (c[ehi] - c[elo])) / np.maximum((ohi - olo) - (ehi - elo), 1)
    return w / np.maximum(fl, np.median(x[x > 0]) * 0.25), w


reps = [_load(f) for f in sorted(glob.glob(f"{REPBW}/*_Cl13_*_H3K9me3_Rep*_*.scale.bw"))]
pooled = np.sum(reps, axis=0)
rp, wp = _ratio(pooled)
n_sup = np.sum([_ratio(x)[0] >= 2 for x in reps], axis=0)          # replicates >= 2x, per bin
locmax = (wp == maximum_filter1d(wp, 21)) & (wp > 0)
REPRO = np.flatnonzero(locmax & (rp >= 4) & (n_sup >= 3))           # reproducible spikes
print(f"[reps] {len(reps)} replicates; reproducible spikes on chr1 (pooled >= 4x, >= 3/4 reps >= 2x): "
      f"{REPRO.size:,}", flush=True)


def rep_scores(peaks):
    """recall of reproducible spikes; share of narrow (<= 3 kb) pieces with replicate support."""
    cov = np.zeros(pooled.size, dtype=bool)
    pieces = []
    for r in peaks:
        if r.chrom != "chr1":
            continue
        cov[r.start // 100: -(-r.end // 100)] = True
        for sp in (r.subpeaks or [r]):
            if sp.end - sp.start <= 3000:
                pieces.append((sp.start // 100, -(-sp.end // 100)))
    rec = float(cov[REPRO].mean()) if REPRO.size else 0.0
    sup = [n_sup[a:b].max() >= 3 for a, b in pieces if b > a]
    return rec, (float(np.mean(sup)) if sup else float("nan")), len(sup)


hdr = (f"{'floor':>5} {'disp':<8} {'fdr':<9} | {'k9 reg':>7} {'k9 sub':>7} {'k9 Mb':>6} "
       f"{'pdcd1':>5} {'repRec':>6} {'repPrec':>7} {'nNarrow':>7} | {'null n':>6} {'null%':>6} | {'narrowF1':>8} {'subF1':>6} | {'mixedF1':>7} {'subF1':>6} | s")
print(hdr)
for floor, disp, fdr in GRID:
    t0 = time.perf_counter()
    over = dict(focal_floor_fraction=floor, focal_dispersion=disp, focal_fdr=fdr)
    res = call_peaks(k9, None, k9_base.replace(**over), return_details=True,
                     segmentation_cache=k9_cache)
    pk = res.peaks
    c, s, e = PDCD1
    loc = [(r.start, r.end) for r in pk if r.chrom == c and r.end > s and r.start < e]
    loc_sub = [(sp.start, sp.end) for r in pk if r.chrom == c and r.end > s and r.start < e
               for sp in (r.subpeaks or [r])]
    row = [len(pk), pk.n_subpeaks, pk.total_bp / 1e6, len(loc_sub)]
    rrec, rprec, nn = rep_scores(pk)
    simres = {}
    for name, (d, base, cache) in sims.items():
        p = call_peaks(d["treat"], None, base.replace(**over), segmentation_cache=cache)
        if name == "null":
            m = evaluate_null(ivs(p), d["genome_size"])
            simres[name] = (len(p), 100 * p.total_bp / d["genome_size"])
        else:
            m = evaluate(ivs(p), d["truth"], d["genome_size"])
            ms = evaluate(sub_ivs(p), d["truth"], d["genome_size"])
            simres[name] = (m["bp_f1"], ms["bp_f1"])
    print(f"{floor:>5} {disp:<8} {fdr:<9} | {row[0]:>7,} {row[1]:>7,} {row[2]:>6.1f} {row[3]:>5} {rrec:>6.3f} {rprec:>7.3f} {nn:>7,} | "
          f"{simres['null'][0]:>6} {simres['null'][1]:>6.2f} | {simres['narrow'][0]:>8.3f} "
          f"{simres['narrow'][1]:>6.3f} | {simres['mixed'][0]:>7.3f} {simres['mixed'][1]:>6.3f} | "
          f"{time.perf_counter() - t0:.0f}", flush=True)
    print(f"      pdcd1 regions {loc}", flush=True)
    print(f"      pdcd1 pieces  {loc_sub}", flush=True)
    print(f"      focal p_cut {res.diagnostics.get('focal', {}).get('p_cut_neglog10')} "
          f"disp {res.diagnostics.get('focal', {}).get('window_dispersion')}", flush=True)
