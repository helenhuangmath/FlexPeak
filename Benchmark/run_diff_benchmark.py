#!/usr/bin/env python3
"""Differential-calling comparison on simulated two-condition data.

    python Benchmark/run_diff_benchmark.py            # everything
    python Benchmark/run_diff_benchmark.py --quick

Datasets (two replicates per condition unless noted, one shared input):

    diff_mixed      narrow + broad gains and losses, equal depth
    diff_depth      the same, condition B sequenced at 50% depth
    diff_single     one replicate per condition
    diff_null       no differential region at all

Comparators: FlexPeak diff (background / TMM normalisation), THOR (rgt-THOR),
MACS2 bdgdiff, SICER2 sicer_df.  Every method is scored from signed intervals
by the same code: base-pair precision/recall/F1 per direction, direction
accuracy (fraction of called base pairs inside truth whose sign is right), and
calls on the null dataset.
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

from flexpeak import load_signal
from flexpeak.peaks import interval_overlap_bp, interval_total_bp
from flexpeak.simulate import simulate_differential, write_bam

import methods as M

warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results_diff")
DATA = os.path.join(HERE, "data_diff")
BIN_SIZE = 25
FRAGMENT_LEN = 50


def dataset_specs(quick: bool) -> Dict[str, Dict]:
    sizes = ({"chr1": 2_000_000, "chr2": 1_000_000} if quick
             else {"chr1": 4_000_000, "chr2": 2_000_000})
    k = 0.5 if quick else 1.0
    base = dict(chrom_sizes=sizes, n_shared_narrow=int(30 * k), n_shared_broad=int(4 * k) or 2,
                n_diff_narrow=int(20 * k), n_diff_broad=int(4 * k) or 2)
    return {
        "diff_mixed": dict(base, seed=201, description="narrow + broad gains and losses"),
        "diff_depth": dict(base, seed=202, depth_b=0.5,
                           description="condition B at 50% depth"),
        "diff_single": dict(base, seed=203, n_reps=1, description="one replicate each"),
        "diff_null": dict(base, seed=204, n_diff_narrow=0, n_diff_broad=0,
                          description="no differential region"),
    }


def build(name: str, spec: Dict) -> Dict:
    os.makedirs(DATA, exist_ok=True)
    spec = dict(spec)
    spec.pop("description", None)
    sim = simulate_differential(bin_size=BIN_SIZE, **spec)
    bams = {}
    for tag, covs in (("A", sim.reps_a), ("B", sim.reps_b), ("input", sim.controls)):
        bams[tag] = []
        for i, cov in enumerate(covs, 1):
            path = os.path.join(DATA, f"{name}_{tag}{i}.bam")
            if not os.path.exists(path):
                print(f"    writing {os.path.basename(path)} ...", flush=True)
                write_bam(path, cov, seed=hash((name, tag, i)) % 10_000)
            bams[tag].append(path)
    t0 = time.perf_counter()
    reps_a = [load_signal(p, bin_size=BIN_SIZE, cache=False) for p in bams["A"]]
    reps_b = [load_signal(p, bin_size=BIN_SIZE, cache=False) for p in bams["B"]]
    ctrl = [load_signal(p, bin_size=BIN_SIZE, cache=False) for p in bams["input"]]
    load_s = time.perf_counter() - t0
    sizes_path = M.chrom_sizes_from_bam(bams["A"][0], os.path.join(DATA, f"{name}.chrom.sizes"))
    return dict(name=name, sim=sim, bams=bams, reps_a=reps_a, reps_b=reps_b, controls=ctrl,
                load_s=load_s, chrom_sizes=sizes_path,
                genome_size=sum(sim.reps_a[0].chrom_sizes.values()))


def score(gain, loss, truth_gain, truth_loss, genome_size) -> Dict[str, float]:
    def prf(calls, truth):
        cb, tb = interval_total_bp(calls), interval_total_bp(truth)
        inter = interval_overlap_bp(calls, truth)
        p = inter / cb if cb else 0.0
        r = inter / tb if tb else float("nan")
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f

    pg, rg, fg = prf(gain, truth_gain)
    pl, rl, fl = prf(loss, truth_loss)
    all_calls, all_truth = gain + loss, truth_gain + truth_loss
    p, r, f = prf(all_calls, all_truth)
    right = interval_overlap_bp(gain, truth_gain) + interval_overlap_bp(loss, truth_loss)
    wrong = interval_overlap_bp(gain, truth_loss) + interval_overlap_bp(loss, truth_gain)
    return dict(n_calls=len(all_calls), called_bp=interval_total_bp(all_calls),
                territory=interval_total_bp(all_calls) / genome_size,
                bp_precision=p, bp_recall=r, bp_f1=f,
                gain_precision=pg, gain_recall=rg, gain_f1=fg,
                loss_precision=pl, loss_recall=rl, loss_f1=fl,
                direction_accuracy=right / (right + wrong) if (right + wrong) else float("nan"))


def methods() -> List[Dict]:
    reg = [
        dict(key="flexpeak_diff_bg", label="FlexPeak diff (background norm)",
             kind="flexpeak", family="FlexPeak", kw=dict(norm="background")),
        dict(key="flexpeak_diff_tmm", label="FlexPeak diff (TMM norm)",
             kind="flexpeak", family="FlexPeak", kw=dict(norm="tmm")),
        dict(key="flexpeak_diff_ctrl", label="FlexPeak diff (background norm, input)",
             kind="flexpeak", family="FlexPeak", kw=dict(norm="background"), control=True),
    ]
    if shutil.which("rgt-THOR"):
        reg += [dict(key="thor", label="THOR (defaults)", kind="thor", family="THOR"),
                dict(key="thor_input", label="THOR (with input)", kind="thor", family="THOR",
                     control=True)]
    macs2 = shutil.which("macs2")
    if macs2:
        reg.append(dict(key="macs2_bdgdiff", label="MACS2 bdgdiff", kind="bdgdiff",
                        family="MACS", exe=macs2))
    if shutil.which("sicer_df") and shutil.which("bedtools"):
        reg.append(dict(key="sicer_df", label="SICER2 sicer_df (W200 G600)", kind="sicer_df",
                        family="SICER2"))
    return reg


def run(spec: Dict, ds: Dict, outdir: str) -> Dict:
    kind = spec["kind"]
    if kind == "flexpeak":
        res = M.run_flexpeak_diff(ds["reps_a"], ds["reps_b"],
                                  ds["controls"] if spec.get("control") else None,
                                  bin_size=100, **spec.get("kw", {}))
        res["runtime_s"] += ds["load_s"]
        return res
    od = os.path.join(outdir, spec["key"], ds["name"])
    if kind == "thor":
        return M.run_thor(ds["bams"]["A"], ds["bams"]["B"], ds["chrom_sizes"], od,
                          name=ds["name"],
                          inputs_a=ds["bams"]["input"] if spec.get("control") else None,
                          inputs_b=ds["bams"]["input"] if spec.get("control") else None)
    if kind == "bdgdiff":
        return M.run_macs2_bdgdiff(spec["exe"], ds["bams"]["A"], ds["bams"]["B"],
                                   ds["bams"]["input"][0], ds["genome_size"], od,
                                   extsize=FRAGMENT_LEN)
    if kind == "sicer_df":
        return M.run_sicer_df(ds["bams"]["A"], ds["bams"]["B"], ds["bams"]["input"][0],
                              ds["genome_size"], od, fragment=FRAGMENT_LEN)
    raise ValueError(kind)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--datasets", default=None)
    args = ap.parse_args(argv)
    os.makedirs(RESULTS, exist_ok=True)
    calls_dir = os.path.join(RESULTS, "calls")
    os.makedirs(calls_dir, exist_ok=True)
    raw = os.path.join(RESULTS, "raw")

    specs = dataset_specs(args.quick)
    if args.datasets:
        want = set(args.datasets.split(","))
        specs = {k: v for k, v in specs.items() if k in want}
    reg = methods()
    print(f"methods: {len(reg)}  datasets: {len(specs)}")
    for m in reg:
        print(f"  - {m['label']}")

    rows, commands = [], []
    for name, spec in specs.items():
        print(f"\n[{name}] {spec['description']}")
        ds = build(name, spec)
        tg, tl = ds["sim"].truth_intervals("gain"), ds["sim"].truth_intervals("loss")
        print(f"    genome {ds['genome_size']:,} bp; truth gain {len(tg)}, loss {len(tl)}")
        for m in reg:
            res = run(m, ds, raw)
            sc = score(res["gain"], res["loss"], tg, tl, ds["genome_size"])
            rows.append(dict(dataset=name, method=m["label"], key=m["key"], family=m["family"],
                             n_truth=len(tg) + len(tl), runtime_s=res["runtime_s"], **sc))
            commands.append(f"[{name}] {m['label']}\n    {res['command']}"
                            + (f"\n    note: {res['note']}" if res.get("note") else ""))
            with open(os.path.join(calls_dir, f"{name}__{m['key']}.bed"), "w") as fh:
                for c, s, e in res["gain"]:
                    fh.write(f"{c}\t{s}\t{e}\tgain\t0\t+\n")
                for c, s, e in res["loss"]:
                    fh.write(f"{c}\t{s}\t{e}\tloss\t0\t-\n")
            if name == "diff_null":
                print(f"    {m['label']:<42} calls {sc['n_calls']:>5}  "
                      f"territory {sc['territory']:6.2%}  ({res['runtime_s']:.1f}s)"
                      + (f"  !! {res['note'][:80]}" if res.get("note", "").startswith("FAILED") else ""))
            else:
                print(f"    {m['label']:<42} calls {sc['n_calls']:>5}  F1 {sc['bp_f1']:5.1%}  "
                      f"gain F1 {sc['gain_f1']:5.1%}  loss F1 {sc['loss_f1']:5.1%}  "
                      f"dir {sc['direction_accuracy']:5.1%}  ({res['runtime_s']:.1f}s)"
                      + (f"  !! {res['note'][:80]}" if res.get("note", "").startswith("FAILED") else ""))

    cols = ["dataset", "method", "key", "family", "n_truth", "n_calls", "called_bp", "territory",
            "bp_precision", "bp_recall", "bp_f1", "gain_precision", "gain_recall", "gain_f1",
            "loss_precision", "loss_recall", "loss_f1", "direction_accuracy", "runtime_s"]
    with open(os.path.join(RESULTS, "metrics.tsv"), "w") as fh:
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
        for exe in ("rgt-THOR", "macs2", "sicer_df", "bedtools"):
            p = shutil.which(exe)
            fh.write(f"{exe:<18} {M.tool_version(p) if p else 'NOT INSTALLED - not run'}\n")
    print(f"\nwrote {os.path.join(RESULTS, 'metrics.tsv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
