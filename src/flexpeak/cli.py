"""Command-line interface, deliberately MACS-shaped so switching is a one-liner."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import __version__

EPILOG = """\
examples:
  # BAM input, self-tuned, nested output
  flexpeak callpeak -t chip.bam -c input.bam -n H3K9me3_sample

  # bigWig input with a preset, no tuning sweep
  flexpeak callpeak -t signal.bw --preset h3k27me3 --no-tune -n K27

  # inspect what the tuner chose and why
  flexpeak tune -t chip.bam --chroms chr1,chr19

  # QC an existing call set
  flexpeak qc -t chip.bam -p out_peaks.nested.tsv

  # basic statistics figures (PDF, Arial) for a call set already on disk
  flexpeak stats -p K27_peaks.nested.tsv --outdir figures
"""


def _add_input_args(p):
    p.add_argument("-t", "--treatment", required=True,
                   help="treatment BAM/CRAM or bigWig")
    p.add_argument("-c", "--control", default=None,
                   help="control/input/IgG BAM or bigWig (optional; no-control "
                        "mode is fully supported)")
    p.add_argument("--chroms", default=None,
                   help="comma-separated chromosome subset")
    p.add_argument("--bin-size", type=int, default=None,
                   help="bin size in bp (default: from preset, else 25)")
    p.add_argument("--mapq", type=int, default=30, help="minimum MAPQ for BAM input")
    p.add_argument("--blacklist", default=None,
                   help="BED of artefact regions to drop calls in (e.g. the "
                        "ENCODE mm10 exclusion list); .bed or .bed.gz")
    p.add_argument("--no-cache", action="store_true",
                   help="do not read/write the coverage cache")
    p.add_argument("--cache-dir", default=None,
                   help="directory for the coverage cache (default: a "
                        ".flexpeak_cache/ beside the input file, which needs the "
                        "input directory to be writable)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="flexpeak",
        description="Flexible, adaptive peak calling (no pretrained model required)",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"flexpeak {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    # -- callpeak ------------------------------------------------------------
    cp = sub.add_parser("callpeak", help="call nested peaks",
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_input_args(cp)
    cp.add_argument("-n", "--name", default="flexpeak", help="output prefix")
    cp.add_argument("--outdir", default=".", help="output directory")
    cp.add_argument("--preset", default=None,
                    help="tf | h3k4me3 | h3k27ac | h3k27me3 | h3k9me3 | h3k36me3 | atac")
    cp.add_argument("--no-tune", action="store_true",
                    help="skip the self-tuning sweep and use preset/defaults "
                         "(required for bit-identical reruns)")
    cp.add_argument("-q", "--qvalue", type=float, default=0.05, help="FDR cutoff")
    cp.add_argument("--min-width", type=int, default=None, help="minimum region width (bp)")
    cp.add_argument("--min-fold", type=float, default=None, help="minimum fold enrichment")
    cp.add_argument("--no-subpeaks", action="store_true",
                    help="skip sub-peak detection (domains only)")
    cp.add_argument("--poisson", action="store_true",
                    help="use a Poisson background instead of negative binomial "
                         "(ablation only; expect inflated significance)")
    cp.add_argument("--background-exclude-fold", type=float, default=None,
                    help="hold bins above this fold out of the local background "
                         "fit (default 2.0; 0 = plain windowed mean, the "
                         "ablation -- expect broad domains to be shredded)")
    cp.add_argument("--segment-bin-size", type=int, default=None,
                    help="bin size the HMM segments at, independent of "
                         "--bin-size (default: the mark's autocorrelation "
                         "length / 8; set equal to --bin-size to segment at "
                         "reporting resolution)")
    cp.add_argument("--no-dwell-floor", action="store_true",
                    help="ablation: let Baum-Welch re-estimate the HMM dwell "
                         "time freely instead of flooring it at the signal's "
                         "autocorrelation length")
    cp.add_argument("--tune-chroms", type=int, default=3,
                    help="number of chromosomes for the tuning sweep")
    cp.add_argument("--figures", action="store_true",
                    help="also write the basic statistics figures (PDF, Arial) "
                         "into <outdir>/figures/ -- requires matplotlib")
    cp.add_argument("--figure-dir", default=None,
                    help="where --figures writes (default: <outdir>/figures)")
    cp.add_argument("--quiet", action="store_true")

    # -- tune ----------------------------------------------------------------
    tp = sub.add_parser("tune", help="run the tuning sweep and report the objective curve")
    _add_input_args(tp)
    tp.add_argument("--preset", default=None, help="starting point for the sweep")
    tp.add_argument("--tune-chroms", type=int, default=3)
    tp.add_argument("--json", action="store_true", help="emit JSON instead of text")

    # -- qc ------------------------------------------------------------------
    qp = sub.add_parser("qc", help="QC a sample, optionally against a call set")
    _add_input_args(qp)
    qp.add_argument("-p", "--peaks", default=None,
                    help="existing FlexPeak nested TSV (else peaks are called first)")
    qp.add_argument("--preset", default=None)
    qp.add_argument("--json", action="store_true")

    # -- stats ---------------------------------------------------------------
    sp = sub.add_parser(
        "stats", help="basic statistics figures (PDF) for an existing call set")
    sp.add_argument("-p", "--peaks", required=True,
                    help="FlexPeak nested TSV written by callpeak "
                         "(<name>_peaks.nested.tsv)")
    sp.add_argument("-n", "--name", default=None,
                    help="output prefix (default: taken from the peaks filename)")
    sp.add_argument("--outdir", default=".", help="output directory")
    sp.add_argument("--title", default=None, help="title for the summary page")
    sp.add_argument("--quiet", action="store_true")

    return p


def _load(args, path, bin_size, chroms):
    from .io.load import load_signal

    kw = {}
    if str(path).lower().endswith((".bam", ".cram", ".sam")):
        kw["mapq"] = args.mapq
    return load_signal(path, bin_size=bin_size, chroms=chroms,
                       cache=not args.no_cache,
                       cache_dir=getattr(args, "cache_dir", None), **kw)


def _resolve_params(args):
    from .caller import CallParams, preset_params

    params = preset_params(args.preset) if getattr(args, "preset", None) else CallParams()
    if getattr(args, "bin_size", None):
        params = params.replace(bin_size=args.bin_size)
    for attr, field in (("qvalue", "qvalue"), ("min_width", "min_width"),
                        ("min_fold", "min_fold")):
        v = getattr(args, attr, None)
        if v is not None:
            params = params.replace(**{field: v})
    if getattr(args, "no_subpeaks", False):
        params = params.replace(call_subpeaks=False)
    if getattr(args, "poisson", False):
        params = params.replace(background_model="poisson")
    bxf = getattr(args, "background_exclude_fold", None)
    if bxf is not None:
        params = params.replace(background_exclude_fold=bxf)
    if getattr(args, "no_dwell_floor", False):
        params = params.replace(dwell_from_autocorrelation=False)
    sbs = getattr(args, "segment_bin_size", None)
    if sbs is not None:
        params = params.replace(segment_bin_size=sbs)
    return params


def cmd_callpeak(args) -> int:
    import os

    from .caller import call_peaks
    from .qc.metrics import qc_metrics, format_qc
    from .tune.sweep import tune

    chroms = args.chroms.split(",") if args.chroms else None
    params = _resolve_params(args)
    say = (lambda *a: None) if args.quiet else (lambda *a: print(*a, file=sys.stderr))

    say(f"[flexpeak] loading treatment: {args.treatment}")
    treat = _load(args, args.treatment, params.bin_size, chroms)
    say(f"[flexpeak]   {treat}")
    ctrl = None
    if args.control:
        say(f"[flexpeak] loading control: {args.control}")
        ctrl = _load(args, args.control, params.bin_size, chroms)

    tune_result = None
    if not args.no_tune:
        say("[flexpeak] self-tuning on a chromosome subset ...")
        tune_result = tune(treat, ctrl, base=params, n_chroms=args.tune_chroms)
        params = tune_result.params
        if not args.quiet:
            print(tune_result.report(), file=sys.stderr)

    blacklist = None
    if args.blacklist:
        from .peaks import read_bed_intervals

        blacklist = read_bed_intervals(args.blacklist)
        say(f"[flexpeak] blacklist: {len(blacklist)} regions from {args.blacklist}")

    say("[flexpeak] calling peaks ...")
    result = call_peaks(treat, ctrl, params, name=args.name, blacklist=blacklist,
                        return_details=True)
    peaks = result.peaks
    if tune_result is not None:
        peaks.params["tuning"] = {
            "chroms": tune_result.chroms,
            "objective": tune_result.score.total,
            "stable": tune_result.stable,
            "notes": tune_result.notes,
            "grid_points": len(tune_result.curve),
        }

    os.makedirs(args.outdir, exist_ok=True)
    prefix = os.path.join(args.outdir, args.name)
    paths = peaks.write_all(prefix)

    m = qc_metrics(treat, peaks, ctrl, result.background)
    with open(f"{prefix}_qc.txt", "w") as fh:
        fh.write(format_qc(m) + "\n")

    if args.figures:
        say("[flexpeak] drawing statistics figures ...")
        figdir = args.figure_dir or os.path.join(args.outdir, "figures")
        try:
            paths.update(_write_figures(peaks, figdir, args.name, args.treatment))
        except (ImportError, ValueError) as e:
            # A missing matplotlib must not throw away a completed call whose
            # outputs are already on disk.
            say(f"[flexpeak] WARNING: figures skipped: {e}")

    dropped = result.diagnostics.get("blacklisted_regions_dropped")
    if dropped:
        say(f"[flexpeak] dropped {dropped} region(s) overlapping the blacklist")
    say(f"[flexpeak] {len(peaks)} regions "
        f"({peaks.n_domains} domains, {peaks.n_subpeaks} subpeaks)")
    for k, v in paths.items():
        say(f"[flexpeak]   {k:<24} {v}")
    say(f"[flexpeak]   {'qc':<24} {prefix}_qc.txt")
    if not treat.is_count:
        say("[flexpeak] NOTE: bigWig input is not raw counts; p-values are approximate.")
    return 0


def _write_figures(peaks, outdir, prefix, source=None, title=None):
    """Draw the statistics figures and return a {label: path} map for logging."""
    import os

    from .plots import peak_stat_figures

    if title is None:
        title = "FlexPeak call statistics"
        if source:
            title += f" -- {os.path.basename(str(source))}"
    figs = peak_stat_figures(peaks, outdir=outdir, prefix=prefix, title=title)
    return {f"fig:{k}": v for k, v in figs.items()}


def cmd_stats(args) -> int:
    import os

    from .peaks import read_nested_tsv

    say = (lambda *a: None) if args.quiet else (lambda *a: print(*a, file=sys.stderr))
    peaks = read_nested_tsv(args.peaks)
    if len(peaks) == 0:
        print(f"flexpeak: error: no regions in {args.peaks}", file=sys.stderr)
        return 1

    name = args.name
    if name is None:
        base = os.path.basename(str(args.peaks))
        for suffix in ("_peaks.nested.tsv.gz", "_peaks.nested.tsv", ".tsv.gz", ".tsv"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        name = base or "flexpeak"

    say(f"[flexpeak] {len(peaks)} regions read from {args.peaks}")
    out = _write_figures(peaks, args.outdir, name,
                         (peaks.params or {}).get("treatment"), title=args.title)
    for k, v in out.items():
        say(f"[flexpeak]   {k:<20} {v}")
    return 0


def cmd_tune(args) -> int:
    from .tune.sweep import tune

    chroms = args.chroms.split(",") if args.chroms else None
    params = _resolve_params(args)
    treat = _load(args, args.treatment, params.bin_size, chroms)
    ctrl = _load(args, args.control, params.bin_size, chroms) if args.control else None

    res = tune(treat, ctrl, base=params, n_chroms=args.tune_chroms)
    if args.json:
        print(json.dumps({
            "params": res.params.to_dict(),
            "chroms": res.chroms,
            "stable": res.stable,
            "notes": res.notes,
            "objective": res.score.total,
            "curve": [{"point": p, "score": s.total, "n_regions": s.n_regions}
                      for p, s in res.curve],
        }, indent=2))
    else:
        print(res.report())
    return 0


def cmd_qc(args) -> int:
    from .caller import call_peaks
    from .peaks import read_nested_tsv
    from .qc.metrics import qc_metrics, format_qc

    chroms = args.chroms.split(",") if args.chroms else None
    params = _resolve_params(args)
    treat = _load(args, args.treatment, params.bin_size, chroms)
    ctrl = _load(args, args.control, params.bin_size, chroms) if args.control else None

    if args.peaks:
        peaks = read_nested_tsv(args.peaks)
        if len(peaks) == 0:
            print(f"flexpeak: error: no regions in {args.peaks}", file=sys.stderr)
            return 1
        m = qc_metrics(treat, peaks, ctrl)
    else:
        result = call_peaks(treat, ctrl, params, return_details=True)
        m = qc_metrics(treat, result.peaks, ctrl, result.background)
    print(json.dumps(m, indent=2, default=str) if args.json else format_qc(m))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    handler = {"callpeak": cmd_callpeak, "tune": cmd_tune, "qc": cmd_qc,
               "stats": cmd_stats}[args.command]
    try:
        return handler(args)
    except (ImportError, ValueError, KeyError, FileNotFoundError) as e:
        print(f"flexpeak: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
