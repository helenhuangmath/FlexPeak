"""Are narrow H3K9me3 spikes reproducible across the 4 Cl13 replicates?

chr1 at 100 bp.  Per replicate and pooled: 200 bp window sum over the mean of
10 kb flanks (window +-200 bp left out).  Candidates: pooled local maxima
(+-1 kb) at >= 3x their flanks.  For each, how many replicates are >= 2x
their own flanks; compared with random background positions.
"""
import glob
import numpy as np
import pyBigWig
from scipy.ndimage import maximum_filter1d

BW = "/vast/projects/wherry/foundation-models-immuno/hhua/data_lab_internal/Huang_SR2025_CnR/geo_processed/per_sample_bw"
files = sorted(glob.glob(f"{BW}/*_Cl13_*_H3K9me3_Rep*_*.scale.bw"))
CH, BS = "chr1", 100


def load(f):
    bw = pyBigWig.open(f)
    n = bw.chroms()[CH] // BS
    v = np.array(bw.stats(CH, 0, n * BS, type="mean", nBins=n), dtype=float)
    bw.close()
    return np.nan_to_num(v)


def ratio(x, k=2, flank=100, guard=2):
    c = np.concatenate([[0.0], np.cumsum(x)])
    n = x.size
    i = np.arange(n)
    lo, hi = np.clip(i - k // 2, 0, n), np.clip(i - k // 2 + k, 0, n)
    w = (c[hi] - c[lo]) / np.maximum(hi - lo, 1)
    elo, ehi = np.clip(lo - guard, 0, n), np.clip(hi + guard, 0, n)
    olo, ohi = np.clip(elo - flank, 0, n), np.clip(ehi + flank, 0, n)
    fl = (c[ohi] - c[olo] - (c[ehi] - c[elo])) / np.maximum((ohi - olo) - (ehi - elo), 1)
    floor = np.median(x[x > 0]) * 0.25
    return w / np.maximum(fl, floor), w


reps = [load(f) for f in files]
print("replicates:", [f.split("/")[-1][:40] for f in files])
pooled = np.sum(reps, axis=0)
rp, wp = ratio(pooled)
rr = [ratio(x)[0] for x in reps]
locmax = (wp == maximum_filter1d(wp, 21)) & (wp > 0)
rng = np.random.default_rng(0)
for thr in (3, 4, 6):
    cand = np.flatnonzero(locmax & (rp >= thr))
    nrep = np.sum([r[cand] >= 2 for r in rr], axis=0)
    pop = np.flatnonzero((rp > 0.8) & (rp < 1.25) & locmax)
    bg = rng.choice(pop, size=min(pop.size, 20000), replace=False)
    nb = np.sum([r[bg] >= 2 for r in rr], axis=0)
    print(f"pooled >= {thr}x: {cand.size:,} spikes on chr1; replicates >= 2x: "
          + " ".join(f"{k}/4={np.mean(nrep == k):.2f}" for k in range(5))
          + f" | background maxima: >=3/4 {np.mean(nb >= 3):.3f}, 4/4 {np.mean(nb == 4):.3f}")
print("\nPdcd1 window chr1:93.94-94.15 Mb, pooled spikes >= 3x:")
s, e = 93_940_000 // BS, 94_152_000 // BS
for j in np.flatnonzero(locmax[s:e] & (rp[s:e] >= 3)) + s:
    print(f"   {j * BS:>12,}  pooled {rp[j]:5.1f}x   reps " + " ".join(f"{r[j]:5.1f}" for r in rr))
