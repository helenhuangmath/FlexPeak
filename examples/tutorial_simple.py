#!/usr/bin/env python3
"""Small self-contained tutorial run on simulated data."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from flexpeak import call_peaks, preset_params
from flexpeak.qc.metrics import format_qc, qc_metrics
from flexpeak.simulate import simulate


def main() -> int:
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tutorial_out")
    os.makedirs(outdir, exist_ok=True)

    data = simulate(
        chrom_sizes={"chrTutorial": 1_200_000},
        bin_size=25,
        n_narrow=12,
        n_broad=2,
        n_mixed=2,
        seed=7,
    )
    params = preset_params("h3k9me3").replace(bin_size=25, min_width=100)
    result = call_peaks(data.treatment, data.control, params,
                        name="tutorial_mixed", return_details=True)

    prefix = os.path.join(outdir, "tutorial_mixed")
    paths = result.peaks.write_all(prefix)
    metrics = qc_metrics(data.treatment, result.peaks, data.control, result.background)
    with open(f"{prefix}_qc.txt", "w") as fh:
        fh.write(format_qc(metrics) + "\n")

    print(format_qc(metrics))
    print("\nOutputs")
    for key, path in paths.items():
        print(f"  {key:<12} {path}")
    print(f"  qc           {prefix}_qc.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
