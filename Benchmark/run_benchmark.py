#!/usr/bin/env python3
"""Method comparison on simulated data with exact ground truth.

    python Benchmark/run_benchmark.py            # everything
    python Benchmark/run_benchmark.py --quick    # smaller genome, faster

Writes:
    Benchmark/results/metrics.tsv     one row per (dataset, method)
    Benchmark/results/versions.txt    tool versions and environment
    Benchmark/results/commands.txt    the exact command line for every run
    Benchmark/results/calls/*.bed     every method's calls, as scored

Then run Benchmark/make_figures.py to render the figures.

Protocol notes
--------------
* All methods see the SAME BAM files and the SAME control, and are scored by the
  same code in metrics.py from call intervals alone.
* Every method is run at its own nominal FDR 5%, which compares methods at
  matched *nominal* error rate.  Matched call count is reported separately by
  the territory column rather than by re-thresholding.
* Ground truth is exact because the data is simulated.  Simulated background is
  drawn from the negative binomial family FlexPeak assumes, which favours
  FlexPeak on the null dataset by construction.  That advantage is stated
  rather than hidden -- see Benchmark/README.md.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
import time
import warnings
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from flexpeak import CallParams, load_signal, preset_params
from flexpeak.simulate import simulate, write_bam

import methods as M
import metrics as MT

warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
CALLS = os.path.join(RESULTS, "calls")
DATA = os.path.join(HERE, "data")

BIN_SIZE = 25
FRAGMENT_LEN = 50


# -- datasets ---------------------------------------------------------------

def dataset_specs(quick: bool) -> Dict[str, Dict]:
    """Four datasets spanning the narrow -> broad axis, plus a null control."""
    sizes = ({"chr1": 1_500_000, "chr2": 750_000} if quick
             else {"chr1": 3_000_000, "chr2": 1_500_000})
    k = 0.5 if quick else 1.0
    return {
        "narrow": dict(
            chrom_sizes=sizes, n_narrow=int(40 * k), n_broad=0, n_mixed=0,
            narrow_fold=8.0, dispersion=0.15, seed=101,
            description="focal peaks only (TF / H3K4me3-like)"),
        "broad": dict(
            chrom_sizes=sizes, n_narrow=0, n_broad=int(6 * k) or 3, n_mixed=0,
            broad_fold=2.5, dispersion=0.15, seed=102,
            description="broad domains only (H3K27me3-like)"),
        "mixed": dict(
            chrom_sizes=sizes, n_narrow=int(30 * k), n_broad=int(4 * k) or 2,
            n_mixed=int(3 * k) or 2, dispersion=0.15, seed=103,
            description="narrow + broad + domains containing sub-peaks (H3K9me3-like)"),
        "lowqual": dict(
            chrom_sizes=sizes, n_narrow=int(40 * k), n_broad=int(3 * k) or 2,
            n_mixed=0, narrow_fold=3.0, broad_fold=1.6,
            background_level=6.0, dispersion=0.40, seed=104,
            description="weak enrichment, overdispersed library (low-quality)"),
        "null": dict(
            chrom_sizes=sizes, n_narrow=0, n_broad=0, n_mixed=0,
            dispersion=0.25, seed=105,
            description="no enrichment anywhere; every call is a false positive"),
    }


def build_dataset(name: str, spec: Dict) -> Dict:
    """Simulate, write BAMs (once, reused across runs), return handles."""
    os.makedirs(DATA, exist_ok=True)
    spec = dict(spec)
    spec.pop("description", None)
    sim = simulate(bin_size=BIN_SIZE, **spec)

    t_bam = os.path.join(DATA, f"{name}_treat.bam")
    c_bam = os.path.join(DATA, f"{name}_control.bam")
    if not (os.path.exists(t_bam) and os.path.exists(c_bam)):
        print(f"    writing BAMs for {name} ...", flush=True)
        write_bam(t_bam, sim.treatment, seed=1)
        write_bam(c_bam, sim.control, seed=2)

    # Load back from the BAM so every method sees the same bytes.  The load is
    # timed and added to FlexPeak's wall clock below: MACS re-parses the BAM on
    # every invocation, so charging FlexPeak nothing for I/O would flatter it.
    t0 = time.perf_counter()
    treat = load_signal(t_bam, bin_size=BIN_SIZE, cache=False)
    ctrl = load_signal(c_bam, bin_size=BIN_SIZE, cache=False)
    load_s = time.perf_counter() - t0

    return {
        "name": name,
        "sim": sim,
        "treat_bam": t_bam,
        "control_bam": c_bam,
        "treat": treat,
        "control": ctrl,
        "load_s": load_s,
        "truth": sim.truth_intervals(),
        "genome_size": sum(sim.treatment.chrom_sizes.values()),
    }


# -- method registry --------------------------------------------------------

def available_methods() -> List[Dict]:
    """Methods to run, in report order.  Unavailable tools are skipped loudly."""
    macs2 = shutil.which("macs2")
    macs3 = shutil.which("macs3")
    reg: List[Dict] = [
        dict(key="flexpeak", label="FlexPeak (defaults)", kind="flexpeak",
             family="FlexPeak"),
        dict(key="flexpeak_tuned", label="FlexPeak (self-tuned)", kind="flexpeak",
             family="FlexPeak", tune=True),
        dict(key="flexpeak_broad", label="FlexPeak (preset h3k27me3)",
             kind="flexpeak", family="FlexPeak", preset="h3k27me3"),
    ]
    if macs2:
        reg += [
            dict(key="macs2_narrow", label="MACS2 narrow", kind="macs", exe=macs2,
                 broad=False, family="MACS"),
            dict(key="macs2_broad", label="MACS2 broad", kind="macs", exe=macs2,
                 broad=True, family="MACS"),
        ]
    if macs3:
        reg += [
            dict(key="macs3_narrow", label="MACS3 narrow", kind="macs", exe=macs3,
                 broad=False, family="MACS"),
            dict(key="macs3_broad", label="MACS3 broad", kind="macs", exe=macs3,
                 broad=True, family="MACS"),
        ]
    # Each re-implemented baseline is run in several configurations -- its
    # defaults and mark-appropriate ones -- because comparing a self-tuned
    # method against fixed-parameter comparators is the strawman that
    # DESIGN.md 10.2 forbids.  Window size is the user's choice in both.
    reg += [
        dict(key="sicer_like_w200", label="SICER/epic2-style, defaults",
             kind="sicer_like", family="baseline", kw=dict(window=200, gap_windows=3)),
        dict(key="sicer_like_tuned", label="SICER/epic2-style, good-faith tuned",
             kind="sicer_like", family="baseline",
             kw=dict(window=200, gap_windows=3, p_eligible=0.05,
                     min_eligible=3, lambda_floor=1.0)),
        dict(key="sicer_like_w1000", label="SICER/epic2-style, 1 kb windows",
             kind="sicer_like", family="baseline", kw=dict(window=1000, gap_windows=3)),
        dict(key="deseq2_like_w200", label="DESeq2/csaw-style NB, 200 bp windows",
             kind="deseq2_like", family="baseline", kw=dict(window=200)),
        dict(key="deseq2_like_w5000", label="DESeq2/csaw-style NB, 5 kb windows",
             kind="deseq2_like", family="baseline", kw=dict(window=5000)),
        dict(key="poisson_ablation", label="FlexPeak, Poisson background (ablation)",
             kind="poisson", family="ablation"),
    ]
    return reg


def run_method(spec: Dict, ds: Dict, outdir: str) -> Dict:
    kind = spec["kind"]
    if kind == "flexpeak":
        params = (preset_params(spec["preset"]).replace(bin_size=BIN_SIZE)
                  if spec.get("preset") else CallParams(bin_size=BIN_SIZE))
        res = M.run_flexpeak(ds["treat"], ds["control"], params,
                             tune_first=spec.get("tune", False))
        # Charge FlexPeak the BAM parse that MACS pays on every run.
        res["runtime_s"] += ds["load_s"]
        return res
    if kind == "macs":
        return M.run_macs(spec["exe"], ds["treat_bam"], ds["control_bam"],
                          ds["genome_size"], os.path.join(outdir, spec["key"]),
                          f"{ds['name']}_{spec['key']}", broad=spec["broad"],
                          extsize=FRAGMENT_LEN)
    if kind == "sicer_like":
        return M.run_sicer_like(ds["treat"], ds["control"], **spec.get("kw", {}))
    if kind == "deseq2_like":
        return M.run_deseq2_like(ds["treat"], ds["control"], **spec.get("kw", {}))
    if kind == "poisson":
        return M.run_poisson_ablation(ds["treat"], ds["control"],
                                      CallParams(bin_size=BIN_SIZE))
    raise ValueError(f"unknown method kind {kind!r}")


# -- main -------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="half-size genome and feature counts")
    ap.add_argument("--datasets", default=None,
                    help="comma-separated subset of dataset names")
    args = ap.parse_args(argv)

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(CALLS, exist_ok=True)
    macs_out = os.path.join(RESULTS, "macs_raw")

    specs = dataset_specs(args.quick)
    if args.datasets:
        want = set(args.datasets.split(","))
        specs = {k: v for k, v in specs.items() if k in want}

    reg = available_methods()
    print(f"methods: {len(reg)}  datasets: {len(specs)}")
    for spec in reg:
        print(f"  - {spec['label']}")
    missing = [t for t in ("epic2", "sicer", "SEACR") if shutil.which(t) is None]
    if missing:
        print(f"  ! not installed, not run: {', '.join(missing)}")

    rows: List[Dict] = []
    commands: List[str] = []

    for name, spec in specs.items():
        print(f"\n[{name}] {spec['description']}")
        ds = build_dataset(name, spec)
        n_truth = len(ds["truth"])
        print(f"    genome {ds['genome_size']:,} bp, truth regions {n_truth}")

        for m in reg:
            res = run_method(m, ds, macs_out)
            ivs = res["intervals"]
            if name == "null":
                score = MT.evaluate_null(ivs, ds["genome_size"])
            else:
                score = MT.evaluate(ivs, ds["truth"], ds["genome_size"])

            row = {"dataset": name, "method": m["label"], "key": m["key"],
                   "family": m["family"], "runtime_s": res["runtime_s"],
                   "n_truth": n_truth, **score}
            rows.append(row)
            commands.append(f"[{name}] {m['label']}\n    {res['command']}"
                            + (f"\n    note: {res['note']}" if res["note"] else ""))

            bed = os.path.join(CALLS, f"{name}__{m['key']}.bed")
            with open(bed, "w") as fh:
                for c, s, e in ivs:
                    fh.write(f"{c}\t{s}\t{e}\n")

            if name == "null":
                print(f"    {m['label']:<44} calls {score['n_calls']:>6}  "
                      f"territory {score['territory']:6.2%}  "
                      f"({res['runtime_s']:.1f}s)")
            else:
                print(f"    {m['label']:<44} calls {score['n_calls']:>6}  "
                      f"F1 {score['bp_f1']:5.1%}  rec {score['region_recall']:5.1%}  "
                      f"fus {score['fusion']:5.1%}  frag {score['fragmentation']:5.1%}  "
                      f"({res['runtime_s']:.1f}s)")

    # -- write outputs ------------------------------------------------------
    cols = ["dataset", "method", "key", "family", "n_truth", "n_calls",
            "called_bp", "territory", "bp_precision", "bp_recall", "bp_f1",
            "jaccard", "region_recall", "fragmentation", "fusion",
            "median_dstart", "median_dend", "n_boundary_matched", "runtime_s"]
    tsv = os.path.join(RESULTS, "metrics.tsv")
    with open(tsv, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(
                "" if r.get(c) is None or (isinstance(r.get(c), float) and np.isnan(r[c]))
                else (f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c]))
                for c in cols) + "\n")

    with open(os.path.join(RESULTS, "commands.txt"), "w") as fh:
        fh.write("\n".join(commands) + "\n")

    import flexpeak
    with open(os.path.join(RESULTS, "versions.txt"), "w") as fh:
        fh.write(f"generated          {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"platform           {platform.platform()}\n")
        fh.write(f"python             {platform.python_version()}\n")
        fh.write(f"flexpeak           {flexpeak.__version__}\n")
        for exe in ("macs2", "macs3"):
            p = shutil.which(exe)
            fh.write(f"{exe:<18} {M.tool_version(p) if p else 'NOT INSTALLED'}\n")
        for exe in ("epic2", "sicer", "SEACR"):
            fh.write(f"{exe:<18} "
                     f"{'installed' if shutil.which(exe) else 'NOT INSTALLED - not run'}\n")
        fh.write("DESeq2 (R)         NOT INSTALLED - not run\n")
        for mod in ("numpy", "scipy", "pysam", "numba"):
            try:
                fh.write(f"{mod:<18} {__import__(mod).__version__}\n")
            except Exception:
                fh.write(f"{mod:<18} unavailable\n")

    print(f"\nwrote {tsv}")
    print(f"      {os.path.join(RESULTS, 'commands.txt')}")
    print(f"      {os.path.join(RESULTS, 'versions.txt')}")
    print(f"      {CALLS}/*.bed")
    print("\nnow run: python Benchmark/make_figures.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
