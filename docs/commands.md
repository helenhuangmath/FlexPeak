# Commands

FlexPeak is used through one program, `flexpeak`, with six commands.
`flexpeak <command> --help` lists every option; the complete list is also in
the {doc}`cli`.

| Command | Purpose |
|---|---|
| `flexpeak callpeak` | Call nested peaks. Self-tunes by default; `--no-tune` uses the defaults or a `--preset`; `--quick` lets the scan choose the scale and reduces the sweep. |
| `flexpeak scan` | Profile a sample at several bin sizes before a full run. |
| `flexpeak diff` | Call differential regions between two conditions, with or without replicates. |
| `flexpeak tune` | Run the tuning sweep only and report the objective curve and the selected parameters. |
| `flexpeak qc` | Library and enrichment quality control for a sample, optionally against an existing call set. |
| `flexpeak stats` | Summary statistics and the full set of figures for an existing call set. |

## callpeak

```bash
flexpeak callpeak -t chip.bam [-c input.bam] -n sample --outdir out/ [options]
```

The main command. Several treatment or control files are pooled. By default
the parameters are chosen by the self-tuning sweep; `--preset` with
`--no-tune` gives a fixed, reproducible parameter set. The options are
described in {doc}`parameters`.

## scan

```bash
flexpeak scan -t chip.bam                       # text report
flexpeak scan -t chip.bam --json -o scan.json   # machine-readable
```

`scan` reads six 10 Mb windows from six chromosomes and profiles them at
several bin sizes. It reports count distributions, the background after
iterative sigma clipping, the signal fraction, and run and gap lengths. It
then classifies the sample as narrow, broad, mixed or null. The class sets
the bin size, background windows, segmentation scale and merging distance
that `callpeak --quick` uses. With `--quick`, `callpeak` then tunes a small
number of parameters on four 25 Mb windows. Options given explicitly on the
command line override the scan.

## diff

```bash
flexpeak diff -a KO_rep1.bam KO_rep2.bam -b WT_rep1.bam WT_rep2.bam -n KO_vs_WT
```

See {doc}`differential`.

## tune

```bash
flexpeak tune -t chip.bam -c input.bam
```

Runs the tuning sweep alone and prints the objective curve, the plateau and
the selected parameters, without writing a call set.

## qc

```bash
flexpeak qc -t chip.bam [-c input.bam] [-p sample_peaks.nested.tsv]
```

Reports the library metrics and quality grade for a sample. When a call set
is given, it also reports FRiP and the call-set statistics.

## stats

```bash
flexpeak stats -p out/sample_peaks.nested.tsv --outdir out/figures
```

Writes summary statistics and every statistics figure for an existing call
set. Requires matplotlib.
