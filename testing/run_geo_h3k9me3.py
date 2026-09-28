#!/usr/bin/env python3
"""One real-data integration test using a public GEO bigWig.

The bigWig is downloaded on demand and kept out of version control. This script
is intentionally small: one public sample, one preset, and structural checks
that catch broken I/O, invalid peak geometry, and unusable QC.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

GEO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE285nnn/GSE285245/suppl/"
    "GSE285245%5Fpool%5FH3K9me3%5FCl13%2Ebw"
)
GEO_NAME = "GSE285245_pool_H3K9me3_Cl13.bw"
MAIN_CHROMS = [f"chr{i}" for i in range(1, 20)] + ["chrX"]

# Figures from this run that the tutorial embeds.  --keep-figures writes them as
# PNG (Markdown cannot render PDF) into docs/figures/tutorial/, which is tracked.
TUTORIAL_FIGURES = ["summary", "width_distribution", "enrichment",
                    "width_vs_enrichment"]
TUTORIAL_FIGDIR = os.path.join(ROOT, "docs", "figures", "tutorial")


def download(url: str, path: str) -> str:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"downloading {url}")
    print(f"       -> {path}")
    urllib.request.urlretrieve(url, path)
    return path


def check(ok: bool, message: str) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  {status} {message}")
    if not ok:
        raise AssertionError(message)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=None,
                    help="existing bigWig path; skips download")
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"),
                    help="download directory, ignored by git")
    ap.add_argument("--outdir", default=os.path.join(HERE, "results", "geo_h3k9me3"))
    ap.add_argument("--cache-dir", default=os.path.join(HERE, "cache"))
    ap.add_argument("--quick", action="store_true",
                    help="chr19 only; omit for chr1-19 plus chrX")
    ap.add_argument("--no-figures", action="store_true",
                    help="skip the statistics figures (they need matplotlib)")
    ap.add_argument("--keep-figures", action="store_true",
                    help="also write the tutorial figures as PNG into "
                         "docs/figures/tutorial/")
    args = ap.parse_args(argv)

    from flexpeak import call_peaks, load_signal, preset_params
    from flexpeak.peaks import read_nested_tsv
    from flexpeak.qc.metrics import qc_metrics, format_qc

    bw = args.input or download(GEO_URL, os.path.join(args.data_dir, GEO_NAME))
    chroms = ["chr19"] if args.quick else MAIN_CHROMS
    label = "chr19" if args.quick else "main"
    name = f"GSE285245_pool_H3K9me3_Cl13_{label}"

    print("FlexPeak GEO real-data test")
    print(f"  source: {GEO_URL}")
    print(f"  input:  {bw}")
    print(f"  scope:  {','.join(chroms)}")

    params = preset_params("h3k9me3")
    cov = load_signal(bw, bin_size=params.bin_size, chroms=chroms,
                      cache=True, cache_dir=args.cache_dir)
    check(cov.n_bins > 0, "coverage loaded")
    check(cov.is_count is False, "bigWig flagged as normalized/non-count signal")

    result = call_peaks(cov, None, params, name=name, return_details=True)
    peaks = result.peaks
    check(len(peaks) > 0, "at least one region called")
    check(all(r.end > r.start for r in peaks), "every region has end > start")
    check(all((a.chrom, a.start) <= (b.chrom, b.start)
              for a, b in zip(peaks.regions, peaks.regions[1:])),
          "regions are sorted")
    overlaps = sum(1 for a, b in zip(peaks.regions, peaks.regions[1:])
                   if a.chrom == b.chrom and b.start < a.end)
    check(overlaps == 0, "no adjacent regions overlap")

    os.makedirs(args.outdir, exist_ok=True)
    prefix = os.path.join(args.outdir, name)
    paths = peaks.write_all(prefix)
    for key in ("narrowPeak", "broadPeak", "bed12", "nested"):
        check(os.path.exists(paths[key]) and os.path.getsize(paths[key]) > 0,
              f"{key} written")

    back = read_nested_tsv(paths["nested"])
    check(len(back) == len(peaks), "nested TSV round-trips region count")
    check(back.n_subpeaks == peaks.n_subpeaks, "nested TSV round-trips subpeak count")

    metrics = qc_metrics(cov, peaks, None, result.background)
    with open(f"{prefix}_qc.txt", "w") as fh:
        fh.write(format_qc(metrics) + "\n")
    check(0.0 < metrics["genome_fraction"] < 0.5, "called territory is sane")
    check(metrics["frip"] >= 0.05, "FRiP is acceptable for this normalized bigWig")

    if not args.no_figures:
        from flexpeak.plots import peak_stat_figures

        title = f"GSE285245 H3K9me3 ({label})"
        figs = peak_stat_figures(peaks, outdir=os.path.join(args.outdir, "figures"),
                                 prefix=name, title=title)
        check(all(os.path.getsize(p) > 0 for p in figs.values()),
              f"{len(figs) - 1} statistics figures written")
        if args.keep_figures:
            kept = peak_stat_figures(peaks, outdir=TUTORIAL_FIGDIR,
                                     prefix=f"geo_h3k9me3_{label}", title=title,
                                     only=TUTORIAL_FIGURES, fmt="png")
            for key, path in kept.items():
                if key != "stats":
                    print(f"  kept {os.path.relpath(path, ROOT)}")

    print(format_qc(metrics))
    print(f"outputs -> {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
