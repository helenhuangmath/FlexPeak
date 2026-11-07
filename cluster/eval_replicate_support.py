"""Score call sets against replicate support, split by tuning vs held-out chromosomes.

    python cluster/eval_replicate_support.py MARK TUNE_CHROMS LABEL=DIR [LABEL=DIR ...]

Per replicate (4 Cl13 per-sample bigWigs, 100 bp): a reference level per
chromosome = median of the 5 kb-smoothed track.  Then

  * call support      a call (each narrowPeak leaf; each broadPeak region) is
                      supported when its mean is >= 1.5x that level in >= 3 of
                      4 replicates;
  * reproducible bp   1 kb tiles >= 2x the level in >= 3 of 4 replicates --
                      what any caller should cover;
  * recall            share of reproducible tiles covered by broadPeak regions;
  * precision proxy   share of called bp (broadPeak) in tiles >= 1.5x in >= 3/4.

Reported separately on the chromosomes the tuning used and on the rest.
"""
import glob
import sys

import numpy as np
import pyBigWig
from scipy.ndimage import uniform_filter1d

BW = "/vast/projects/wherry/foundation-models-immuno/hhua/data_lab_internal/Huang_SR2025_CnR/geo_processed/per_sample_bw"
BS, TILE = 100, 10
mark, tune_chroms = sys.argv[1], set(sys.argv[2].split(","))
sets = [a.split("=", 1) for a in sys.argv[3:]]
CHROMS = ["chr1", "chr4", "chr6", "chr10", "chr11", "chr15"]
files = sorted(glob.glob(f"{BW}/*_Cl13_*_{mark}_Rep*_*.scale.bw"))
assert len(files) == 4, files


def load(f, c):
    bw = pyBigWig.open(f)
    n = bw.chroms()[c] // BS
    v = np.nan_to_num(np.asarray(bw.values(c, 0, n * BS, numpy=True), dtype=np.float32))
    bw.close()
    return v.reshape(-1, BS).mean(axis=1)


reps = {c: [load(f, c) for f in files] for c in CHROMS}
level = {c: [max(float(np.median(uniform_filter1d(x, 50))), 1e-3) for x in reps[c]] for c in CHROMS}
tiles = {}
for c in CHROMS:
    nt = reps[c][0].size // TILE
    t = np.array([x[: nt * TILE].reshape(nt, TILE).mean(axis=1) / lv
                  for x, lv in zip(reps[c], level[c])])
    tiles[c] = ((t >= 2.0).sum(axis=0) >= 3, (t >= 1.5).sum(axis=0) >= 3)


def read(path):
    out = {}
    for line in open(path):
        if line.startswith(("#", "track")):
            continue
        f = line.split("\t")
        if f[0] in reps:
            out.setdefault(f[0], []).append((int(f[1]), int(f[2])))
    return out


def supported(c, s, e):
    b0, b1 = s // BS, max(s // BS + 1, -(-e // BS))
    return sum(float(x[b0:b1].mean()) >= 1.5 * lv for x, lv in zip(reps[c], level[c])) >= 3


print(f"\n## {mark}: replicate support (tuning chroms {sorted(tune_chroms)})")
print(f"  {'set':<12} {'group':<9} {'regions':>8} {'Mb':>7} {'leaves':>8} {'leaf sup':>8} "
      f"{'reg sup':>8} {'recall':>7} {'prec':>6}")
for label, d in sets:
    bp_ = read(glob.glob(f"{d}/*_peaks.broadPeak")[0])
    np_ = read(glob.glob(f"{d}/*_peaks.narrowPeak")[0])
    for grp, keep in (("tuning", lambda c: c in tune_chroms), ("held-out", lambda c: c not in tune_chroms)):
        cs = [c for c in CHROMS if keep(c)]
        n_r = sum(len(bp_.get(c, [])) for c in cs)
        mb = sum(e - s for c in cs for s, e in bp_.get(c, [])) / 1e6
        leaves = [(c, s, e) for c in cs for s, e in np_.get(c, [])]
        regs = [(c, s, e) for c in cs for s, e in bp_.get(c, [])]
        ls = np.mean([supported(*x) for x in leaves]) if leaves else float("nan")
        rs = np.mean([supported(*x) for x in regs]) if regs else float("nan")
        cov_rep = tot_rep = cov_bp = sup_bp = 0
        for c in cs:
            rep2, rep15 = tiles[c]
            m = np.zeros(rep2.size, bool)
            for s, e in bp_.get(c, []):
                m[s // (BS * TILE): -(-e // (BS * TILE))] = True
            cov_rep += int((m & rep2).sum()); tot_rep += int(rep2.sum())
            cov_bp += int(m.sum()); sup_bp += int((m & rep15).sum())
        print(f"  {label:<12} {grp:<9} {n_r:>8,} {mb:>7.1f} {len(leaves):>8,} {ls:>8.3f} {rs:>8.3f} "
              f"{cov_rep / max(tot_rep, 1):>7.3f} {sup_bp / max(cov_bp, 1):>6.3f}")

LOCI = {"Pdcd1": ("chr1", 93_938_305, 94_152_553), "Havcr2": ("chr11", 46_414_000, 46_517_000),
        "Tcf7": ("chr11", 52_212_000, 52_323_000), "Tox": ("chr4", 6_586_000, 7_099_000)}
for label, d in sets:
    bp_ = read(glob.glob(f"{d}/*_peaks.broadPeak")[0])
    for name, (c, a, b) in LOCI.items():
        r = [(s, e) for s, e in bp_.get(c, []) if e > a and s < b]
        print(f"  {label:<12} {name:<7} {len(r):>3} regions, {sum(min(e, b) - max(s, a) for s, e in r) / 1e3:7.1f} kb: "
              + ", ".join(f"{s / 1e6:.3f}-{e / 1e6:.3f}" for s, e in r[:14]))
