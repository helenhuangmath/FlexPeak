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

  # quick scan of chr1 at 10/25/50 bp: distributions, background, recommended settings
  flexpeak scan -t chip.bam

  # scan first, then tune a small grid from its recommendation (fastest path)
  flexpeak callpeak -t chip.bam -c input.bam -n sample --quick

  # inspect what the tuner chose and why
  flexpeak tune -t chip.bam --chroms chr1,chr19

  # differential regions between two conditions, THOR-style, with replicates
  flexpeak diff -a a_rep1.bam a_rep2.bam -b b_rep1.bam b_rep2.bam -n A_vs_B

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
                   help="comma-separated chromosome subset (default: the "
                        "assembled chromosomes -- unplaced / random / alt "
                        "contigs, chrM and EBV are left out)")
    p.add_argument("--all-contigs", action="store_true",
                   help="also call on unplaced, random and alt contigs "
                        "(everything in the file except chrM and EBV)")
    p.add_argument("--bin-size", type=int, default=None,
                   help="bin size in bp (default: from preset, else 25)")
    p.add_argument("--mapq", type=int, default=30, help="minimum MAPQ for BAM input")
    p.add_argument("--threads", type=int, default=0,
                   help="worker processes, one chromosome each, for reading a "
                        "BAM and for the peak call itself (default 0: automatic "
                        "-- every CPU allocated to the job for the call; up to 8 "
                        "for a large BAM, serial for a small one)")
    p.add_argument("--blacklist", default=None,
                   help="BED of artefact regions to drop calls in (e.g. the "
                        "ENCODE mm10 exclusion list); .bed or .bed.gz")
    p.add_argument("--tss", default=None, metavar="GTF|BED",
                   help="gene annotation (GTF/GFF) or TSS BED: adds the TSS "
                        "enrichment score to the QC report (the ATAC "
                        "signal-to-background metric)")
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
    cp.add_argument("--dispersion", default=None, metavar="A0,A1",
                    help="pin the per-bin dispersion trend alpha = a0 + a1/mu "
                         "instead of fitting it on this sample -- e.g. the median "
                         "over the libraries of one experiment (see <name>_qc.json), "
                         "so replicates are tested under one noise model")
    cp.add_argument("--segment-dispersion", default=None, metavar="A0,A1",
                    help="pin the dispersion trend of the segmentation-bin sums "
                         "(what the HMM segments with) the same way")
    cp.add_argument("--max-width", type=int, default=None,
                    help="widest region reported, bp; a wider one is reported as "
                         "its sub-peaks (default: 50000 with --preset atac, else "
                         "no cap; 0 = no cap)")
    cp.add_argument("--windows", default=None,
                    help="comma-separated local-background windows in bp, e.g. "
                         "1000,5000,10000 (MACS) or 1000000,5000000 (domain marks); "
                         "overrides the preset and the quick scan")
    cp.add_argument("--no-focal", action="store_true",
                    help="skip the focal pass (MACS-style 5/10 kb local background "
                         "segmented at fine scale, whose regions become the sub-peaks "
                         "of domains); the scale-space sub-peak search runs instead")
    cp.add_argument("--focal-scales", default=None,
                    help="comma-separated sliding-window widths (bp) for the focal "
                         "test, e.g. 200,800,3000 (default: from --quick's scan of the "
                         "run-width distribution, else one window)")
    cp.add_argument("--focal-background", choices=("flank", "plain"), default=None,
                    help="focal background per window: 'flank' (default; local mean "
                         "of the flanks with the window left out, floored at the "
                         "median background) or 'plain' (previous: 10 kb mean incl. "
                         "the peak, maxed with the wide background)")
    cp.add_argument("--segment-bin-size", type=int, default=None,
                    help="bin size the HMM segments at, independent of "
                         "--bin-size (default: the mark's autocorrelation "
                         "length / 8; set equal to --bin-size to segment at "
                         "reporting resolution)")
    cp.add_argument("--hmm-fit-fraction", type=float, default=None,
                    help="share of each chromosome the HMM's EM iterations run "
                         "on (evenly spaced blocks); the posterior is always "
                         "computed on the whole chromosome (default 1.0; 0.25 "
                         "with --preset atac)")
    cp.add_argument("--no-dwell-floor", action="store_true",
                    help="ablation: let Baum-Welch re-estimate the HMM dwell "
                         "time freely instead of flooring it at the signal's "
                         "autocorrelation length")
    cp.add_argument("--tune-chroms", type=int, default=None,
                    help="number of whole chromosomes for the tuning sweep (default 3; "
                         "with --quick the sweep runs on four stratified 25 Mb "
                         "windows instead, unless this is given)")
    cp.add_argument("--replicates", nargs="+", default=None, metavar="FILE",
                    help="replicate BAM/bigWig files of the treatment.  The tuning "
                         "halves become replicate groups (1,3,.. vs 2,4,..) instead "
                         "of read splits, so settings are scored on agreement "
                         "between independent replicates; loaded on the tuning "
                         "chromosomes only")
    cp.add_argument("--quick", action="store_true",
                    help="run the quick scan first (stratified windows over several "
                         "chromosomes, several bin sizes) and let it choose the bin "
                         "size, windows, scale and a small one-axis-at-a-time "
                         "tuning sweep instead of the full 27-point grid; "
                         "--preset/--bin-size still win where given")
    cp.add_argument("--scan-chrom", default=None,
                    help="scan this one chromosome instead of the stratified "
                         "windows")
    cp.add_argument("--scan-windows", type=int, default=6,
                    help="chromosomes the quick scan samples, one window each, "
                         "60 Mb in total (default 6; 1 = the largest chromosome)")
    cp.add_argument("--scan-bins", default="10,25,50",
                    help="bin sizes the quick scan profiles (default 10,25,50)")
    cp.add_argument("--scan-sd", type=float, default=5.0,
                    help="bins beyond this many SD of the clipped background "
                         "are candidate signal in the quick scan (default 5)")
    cp.add_argument("--keep-cache", action="store_true",
                    help="keep the binned-coverage cache after the run, so that "
                         "calling the same input again skips reading it (by "
                         "default the cache is deleted when the run ends)")
    cp.add_argument("--figures", nargs="?", const="qc", default=None,
                    choices=("qc", "all"),
                    help="write the QC figure <name>_qc.pdf: one page with the "
                         "call summary, width and enrichment distributions, "
                         "per-chromosome counts and, where available, the "
                         "fragment-size ladder and the TSS profile.  "
                         "'--figures all' writes every statistics figure into "
                         "<outdir>/figures/ instead.  Requires matplotlib")
    cp.add_argument("--figure-dir", default=None,
                    help="where --figures writes (default: <outdir>, or "
                         "<outdir>/figures for 'all')")
    cp.add_argument("--quiet", action="store_true")

    # -- tune ----------------------------------------------------------------
    tp = sub.add_parser("tune", help="run the tuning sweep and report the objective curve")
    _add_input_args(tp)
    tp.add_argument("--preset", default=None, help="starting point for the sweep")
    tp.add_argument("--tune-chroms", type=int, default=3)
    tp.add_argument("--replicates", nargs="+", default=None, metavar="FILE",
                    help="replicate files: tuning halves become replicate groups")
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

    # -- scan ----------------------------------------------------------------
    sc = sub.add_parser(
        "scan", help="quick quantification of one chromosome at several bin "
                     "sizes: distributions, clipped background, recommended "
                     "settings")
    _add_input_args(sc)
    sc.add_argument("--scan-chrom", default=None,
                    help="profile this one chromosome instead of stratified windows")
    sc.add_argument("--scan-windows", type=int, default=6,
                    help="chromosomes to sample, one window each, 60 Mb in total "
                         "(default 6)")
    sc.add_argument("--scan-bins", default="10,25,50",
                    help="comma-separated bin sizes (default 10,25,50)")
    sc.add_argument("--scan-sd", type=float, default=5.0,
                    help="SD multiple that separates candidate signal from "
                         "background (default 5)")
    sc.add_argument("--json", action="store_true", help="emit JSON instead of text")
    sc.add_argument("-o", "--out", default=None, help="also write the JSON here")

    # -- diff ----------------------------------------------------------------
    dp = sub.add_parser(
        "diff", help="differential regions between two conditions (THOR-style "
                     "HMM over the two-condition signal, replicates supported)")
    dp.add_argument("-a", "--cond-a", nargs="+", required=True, metavar="FILE",
                    help="condition A BAM/bigWig files (replicates)")
    dp.add_argument("-b", "--cond-b", nargs="+", required=True, metavar="FILE",
                    help="condition B BAM/bigWig files (replicates)")
    dp.add_argument("--control-a", nargs="*", default=None, metavar="FILE",
                    help="input/IgG for condition A (optional)")
    dp.add_argument("--control-b", nargs="*", default=None, metavar="FILE",
                    help="input/IgG for condition B (optional)")
    dp.add_argument("-n", "--name", default="flexpeak_diff", help="output prefix")
    dp.add_argument("--outdir", default=".", help="output directory")
    dp.add_argument("--chroms", default=None, help="comma-separated chromosome subset")
    dp.add_argument("--bin-size", type=int, default=None,
                    help="bin size for the HMM (default: from the quick scan, "
                         "else 100 bp)")
    dp.add_argument("--norm", default="background",
                    choices=("background", "tmm", "total"),
                    help="library normalisation: background (match the clipped "
                         "background level; default), tmm (trimmed mean of "
                         "M-values over informative bins), total (library size)")
    dp.add_argument("-q", "--qvalue", type=float, default=0.05, help="FDR cutoff")
    dp.add_argument("--min-fold", type=float, default=1.5,
                    help="minimum |fold change| between conditions (default 1.5)")
    dp.add_argument("--min-count", type=float, default=10.0,
                    help="minimum normalised counts in a region, summed over "
                         "conditions (default 10)")
    dp.add_argument("--min-width", type=int, default=None,
                    help="minimum region width (default 2 bins)")
    dp.add_argument("--posterior-cutoff", type=float, default=0.5)
    dp.add_argument("--rho", type=float, default=None,
                    help="fix the beta-binomial overdispersion (default: from "
                         "replicates, else EM floored at 0.02)")
    dp.add_argument("--within-peaks", action="store_true",
                    help="restrict differential calls to the union of FlexPeak "
                         "peaks called on each pooled condition")
    dp.add_argument("--no-scan", action="store_true",
                    help="skip the quick scan that picks the bin size and scale")
    dp.add_argument("--mapq", type=int, default=30)
    dp.add_argument("--threads", type=int, default=0)
    dp.add_argument("--no-cache", action="store_true")
    dp.add_argument("--cache-dir", default=None)
    dp.add_argument("--quiet", action="store_true")

    return p


def _load(args, path, bin_size, chroms):
    from .io.load import load_signal

    kw = {}
    if str(path).lower().endswith((".bam", ".cram", ".sam")):
        kw["mapq"] = args.mapq
        kw["n_jobs"] = getattr(args, "threads", 0)
    return load_signal(path, bin_size=bin_size, chroms=chroms,
                       cache=not args.no_cache,
                       cache_dir=getattr(args, "cache_dir", None),
                       primary_only=not getattr(args, "all_contigs", False), **kw)


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
    hff = getattr(args, "hmm_fit_fraction", None)
    if hff is not None:
        params = params.replace(hmm_fit_fraction=hff)
    if getattr(args, "no_dwell_floor", False):
        params = params.replace(dwell_from_autocorrelation=False)
    sbs = getattr(args, "segment_bin_size", None)
    if sbs is not None:
        params = params.replace(segment_bin_size=sbs)
    for attr in ("dispersion", "segment_dispersion"):
        v = getattr(args, attr, None)
        if v:
            pair = tuple(float(x) for x in str(v).split(","))
            if len(pair) != 2:
                raise SystemExit(f"flexpeak: error: --{attr.replace('_', '-')} takes A0,A1")
            params = params.replace(**{attr: pair})
    mw = getattr(args, "max_width", None)
    if mw is not None:
        params = params.replace(max_width=mw or None)
    win = getattr(args, "windows", None)
    if win:
        params = params.replace(windows=tuple(int(w) for w in str(win).split(",") if w.strip()))
    if getattr(args, "no_focal", False):
        params = params.replace(focal_pass=False)
    fs = getattr(args, "focal_scales", None)
    if fs:
        params = params.replace(focal_scales=tuple(int(w) for w in str(fs).split(",") if w.strip()))
    fb = getattr(args, "focal_background", None)
    if fb:
        params = params.replace(focal_background=fb)
    return params


def cmd_callpeak(args) -> int:
    """``callpeak``, then remove the coverage caches it read or wrote.

    The cache only pays off when the same input is called again; after a
    finished run it is disk space (tens of MB per library at 10 bp).  It is
    removed whether the run succeeded or failed, unless ``--keep-cache``.
    """
    try:
        return _callpeak(args)
    finally:
        if not args.no_cache and not getattr(args, "keep_cache", False):
            _remove_run_caches(args)


def _remove_run_caches(args) -> None:
    from .io.load import remove_caches

    inputs = [args.treatment, args.control] + list(getattr(args, "replicates", None) or [])
    n = 0
    for f in inputs:
        if f:
            n += len(remove_caches(f, args.cache_dir))
    if n and not args.quiet:
        print(f"[flexpeak] removed {n} coverage cache file(s) "
              f"(--keep-cache keeps them for a rerun)", file=sys.stderr)


def _callpeak(args) -> int:
    import os

    from .caller import call_peaks
    from .qc.metrics import qc_metrics, format_qc
    from .tune.sweep import tune

    chroms = args.chroms.split(",") if args.chroms else None
    params = _resolve_params(args)
    say = (lambda *a: None) if args.quiet else (lambda *a: print(*a, file=sys.stderr))
    tss_future = _start_tss_reader(args)

    grid = None
    scan_rec = None
    if getattr(args, "quick", False):
        from .scan import scan_file

        say(f"[flexpeak] quick scan of {args.treatment} ...")
        prof = scan_file(args.treatment, bin_sizes=_scan_bins(args),
                         chrom=args.scan_chrom, n_sd=args.scan_sd,
                         n_windows=args.scan_windows,
                         cache=not args.no_cache, cache_dir=args.cache_dir,
                         **_load_kw(args, args.treatment))
        scan_rec = prof.recommend()
        if not args.quiet:
            print(prof.report(scan_rec), file=sys.stderr)
        over = scan_rec.overrides()
        # Explicit user choices win over the scan.
        if args.preset:
            for k in ("windows", "min_width", "domain_min_width", "posterior_cutoff",
                      "min_fold", "bin_size"):
                over.pop(k, None)
        if args.bin_size:
            over.pop("bin_size", None)
        if getattr(args, "windows", None):
            over.pop("windows", None)
        for attr, key in (("min_width", "min_width"), ("min_fold", "min_fold"),
                          ("segment_bin_size", "segment_bin_size"),
                          ("focal_scales", "focal_scales")):
            if getattr(args, attr, None) is not None:
                over.pop(key, None)
        params = params.replace(**over)
        grid = scan_rec.grid
        say(f"[flexpeak]   scan: {scan_rec.signal_class} mark, bin {params.bin_size} bp, "
            f"windows {list(params.windows)}, focal windows {list(params.focal_scales)} bp, "
            f"max_gap {params.max_gap} bp "
            f"({prof.elapsed_s:.1f} s)")

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
        from .tune.sweep import pick_tuning_chroms, tuning_windows

        quick = getattr(args, "quick", False)
        t_windows = None
        if quick and not args.tune_chroms:
            # Stratified windows over four chromosomes, 100 Mb in total.
            t_windows = tuning_windows(treat)
            t_chroms = sorted({c for c, _, _ in t_windows})
        else:
            t_chroms = pick_tuning_chroms(treat, args.tune_chroms or 3)
        reps = None
        if args.replicates:
            say(f"[flexpeak] loading {len(args.replicates)} replicates on "
                f"{', '.join(t_chroms)} for tuning")
            reps = [_load(args, f, params.bin_size, t_chroms) for f in args.replicates]
        tune_result = tune(treat, ctrl, base=params, chroms=None if t_windows else t_chroms,
                           grid=grid, replicates=reps, windows=t_windows)
        params = tune_result.params
        if not args.quiet:
            print(tune_result.report(), file=sys.stderr)

    blacklist = None
    if args.blacklist:
        from .peaks import read_bed_intervals

        blacklist = read_bed_intervals(args.blacklist)
        say(f"[flexpeak] blacklist: {len(blacklist)} regions from {args.blacklist}")

    import time as _time

    jobs = getattr(args, "threads", 0)
    say("[flexpeak] calling peaks ...")
    _t0 = _time.perf_counter()
    result = call_peaks(treat, ctrl, params, name=args.name, blacklist=blacklist,
                        return_details=True, n_jobs=getattr(args, "threads", 0))
    peaks = result.peaks
    tm = result.diagnostics.get("timing") or {}
    if tm:
        say(f"[flexpeak]   called in {_time.perf_counter() - _t0:.1f} s ("
            + ", ".join(f"{k} {v:.1f}" for k, v in tm.items()) + ")")
    if scan_rec is not None:
        peaks.params["scan"] = {
            "class": scan_rec.signal_class, "scale_bp": scan_rec.scale_bp,
            "chrom": prof.chrom, "bin_sizes": prof.bin_sizes, "n_sd": prof.n_sd,
            "regions": [f"{c}:{a}-{b}" for c, a, b in prof.regions],
        }
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

    # One QC report per run: library and call-set statistics together, as
    # text and as JSON.  (``flexpeak stats`` still writes the separate
    # summary and the full figure set for a call set on disk.)
    import json

    from .plots.stats import format_stats, peak_stats

    _t1 = _time.perf_counter()
    m = qc_metrics(treat, peaks, ctrl, result.background, **_qc_kw(args, tss_future))
    m["timing_s"] = {k: round(v, 2) for k, v in tm.items()}
    calls = peak_stats(peaks) if len(peaks) else None
    with open(f"{prefix}_qc.txt", "w") as fh:
        fh.write(format_qc(m) + "\n")
        if calls:
            fh.write("\n" + format_stats(calls) + "\n")
    with open(f"{prefix}_qc.json", "w") as fh:
        json.dump(dict(m, calls=calls), fh, indent=2, default=str)
        fh.write("\n")
    paths["qc"] = f"{prefix}_qc.txt"
    paths["qc_json"] = f"{prefix}_qc.json"
    say(f"[flexpeak]   outputs written in {_t1 - _t0 - sum(tm.values()):.1f} s, "
        f"QC {_time.perf_counter() - _t1:.1f} s")

    if args.figures:
        try:
            if args.figures == "all":
                say("[flexpeak] drawing every statistics figure ...")
                figdir = args.figure_dir or os.path.join(args.outdir, "figures")
                paths.update(_write_figures(peaks, figdir, args.name, args.treatment))
            else:
                from .plots.stats import qc_figure

                figdir = args.figure_dir or args.outdir
                title = f"FlexPeak QC -- {args.name}"
                paths["fig:qc"] = qc_figure(
                    peaks, os.path.join(figdir, f"{args.name}_qc.pdf"), title=title,
                    qc=dict(m, fragment_sizes=treat.meta.get("fragment_sizes")))
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
    if not treat.is_count:
        say("[flexpeak] NOTE: bigWig input is not raw counts; p-values are approximate.")
    return 0


def _write_summary(peaks, prefix):
    """Counts and width distribution of a call set: ``<prefix>_summary.txt``
    (human-readable, with width histograms) and ``<prefix>_summary.json``."""
    import json

    from .plots.stats import format_stats, peak_stats

    s = peak_stats(peaks)
    with open(f"{prefix}_summary.txt", "w") as fh:
        fh.write(format_stats(s) + "\n")
    with open(f"{prefix}_summary.json", "w") as fh:
        json.dump(s, fh, indent=2, default=str)
        fh.write("\n")
    return {"summary": f"{prefix}_summary.txt", "summary_json": f"{prefix}_summary.json"}


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
    os.makedirs(args.outdir, exist_ok=True)
    out = _write_summary(peaks, os.path.join(args.outdir, name))
    if not args.quiet:
        with open(out["summary"]) as fh:
            print(fh.read(), file=sys.stderr)
    out.update(_write_figures(peaks, args.outdir, name,
                         (peaks.params or {}).get("treatment"), title=args.title))
    for k, v in out.items():
        say(f"[flexpeak]   {k:<20} {v}")
    return 0


def cmd_tune(args) -> int:
    from .tune.sweep import tune

    chroms = args.chroms.split(",") if args.chroms else None
    params = _resolve_params(args)
    treat = _load(args, args.treatment, params.bin_size, chroms)
    ctrl = _load(args, args.control, params.bin_size, chroms) if args.control else None

    from .tune.sweep import pick_tuning_chroms

    t_chroms = pick_tuning_chroms(treat, args.tune_chroms)
    reps = ([_load(args, f, params.bin_size, t_chroms) for f in args.replicates]
            if args.replicates else None)
    res = tune(treat, ctrl, base=params, chroms=t_chroms, replicates=reps)
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
        m = qc_metrics(treat, peaks, ctrl, **_qc_kw(args))
    else:
        result = call_peaks(treat, ctrl, params, return_details=True,
                            n_jobs=getattr(args, "threads", 0))
        m = qc_metrics(treat, result.peaks, ctrl, result.background, **_qc_kw(args))
    print(json.dumps(m, indent=2, default=str) if args.json else format_qc(m))
    return 0


def _start_tss_reader(args):
    """Begin parsing ``--tss`` in a side process and return its future.

    A genome annotation is a 1-2 GB text file and parsing it is several
    seconds of pure waiting at the end of the run; started here, before
    anything large is in memory, it is done long before the QC asks for it.
    None when there is nothing to read or no ``fork`` to do it with.
    """
    import multiprocessing as mp

    if not getattr(args, "tss", None) or "fork" not in mp.get_all_start_methods():
        return None
    from concurrent.futures import ProcessPoolExecutor

    from .qc.metrics import read_tss

    pool = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("fork"))
    fut = pool.submit(read_tss, args.tss)
    pool.shutdown(wait=False)
    return fut


def _qc_kw(args, tss_future=None):
    """Assay-specific QC inputs: only what was asked for, so the default
    report is unchanged."""
    kw = {}
    preset = (getattr(args, "preset", None) or "").lower()
    if preset == "atac":
        kw["assay"] = "atac"
    tss = getattr(args, "tss", None)
    if tss:
        from .qc.metrics import read_tss

        kw["tss"] = tss_future.result() if tss_future is not None else read_tss(tss)
    return kw


def _scan_bins(args):
    return tuple(int(x) for x in str(args.scan_bins).split(",") if x.strip())


def _load_kw(args, path):
    kw = {}
    if str(path).lower().endswith((".bam", ".cram", ".sam")):
        kw["mapq"] = args.mapq
        kw["n_jobs"] = getattr(args, "threads", 0)
    return kw


def cmd_scan(args) -> int:
    from .scan import scan_file

    prof = scan_file(args.treatment, bin_sizes=_scan_bins(args), chrom=args.scan_chrom,
                     n_sd=args.scan_sd, n_windows=args.scan_windows,
                     cache=not args.no_cache,
                     cache_dir=args.cache_dir, **_load_kw(args, args.treatment))
    rec = prof.recommend()
    if args.out:
        prof.to_json(args.out, rec)
    print(prof.to_json(rec=rec) if args.json else prof.report(rec))
    return 0


def cmd_diff(args) -> int:
    import os

    from .diff import DiffParams, call_differential

    say = (lambda *a: None) if args.quiet else (lambda *a: print(*a, file=sys.stderr))
    chroms = args.chroms.split(",") if args.chroms else None
    params = DiffParams(
        qvalue=args.qvalue, min_fold=args.min_fold, min_count=args.min_count,
        posterior_cutoff=args.posterior_cutoff, norm=args.norm,
        within_peaks=args.within_peaks,
    )
    if args.bin_size:
        params = params.replace(bin_size=args.bin_size)
    if args.min_width is not None:
        params = params.replace(min_width=args.min_width)
    if args.rho is not None:
        params = params.replace(rho=args.rho)

    def loader(path, bin_size):
        from .io.load import load_signal

        return load_signal(path, bin_size=bin_size, chroms=chroms,
                           cache=not args.no_cache, cache_dir=args.cache_dir,
                           **_load_kw(args, path))

    res = call_differential(
        args.cond_a, args.cond_b, controls_a=args.control_a or None,
        controls_b=args.control_b or None, params=params, loader=loader,
        name=args.name, scan=not args.no_scan, log=say,
    )
    os.makedirs(args.outdir, exist_ok=True)
    prefix = os.path.join(args.outdir, args.name)
    paths = res.write_all(prefix)
    say(f"[flexpeak diff] {len(res.regions)} differential regions "
        f"({res.n_gain} gain in A, {res.n_loss} loss in A)")
    for k, v in paths.items():
        say(f"[flexpeak diff]   {k:<16} {v}")
    print(res.report(), file=sys.stderr)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    handler = {"callpeak": cmd_callpeak, "tune": cmd_tune, "qc": cmd_qc,
               "stats": cmd_stats, "scan": cmd_scan, "diff": cmd_diff}[args.command]
    try:
        return handler(args)
    except (ImportError, ValueError, KeyError, FileNotFoundError) as e:
        print(f"flexpeak: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
