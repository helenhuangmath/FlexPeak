# Input and output

## Input

BAM or CRAM
: Coordinate-sorted and indexed. Paired-end reads are converted to fragments
  (one per proper pair, template length ≤ 1 kb). Single-end reads are extended
  by a fragment length estimated from strand cross-correlation. Duplicate,
  secondary, supplementary, QC-fail and unmapped reads are excluded. The
  default minimum mapping quality is 30 (`--mapq`).

bigWig
: Treated as normalised signal and rescaled to pseudo-counts. The resulting
  p- and q-values are approximate, and the output header says so.

Chromosomes
: By default only assembled chromosomes are analysed. Unplaced, random and
  alternate contigs, the mitochondrial genome and EBV are excluded.
  `--chroms` selects a subset and `--all-contigs` includes every sequence.

Exclusion list
: `--blacklist` takes a BED file (optionally gzipped); calls inside it are
  discarded.

Coverage cache
: During a run, binned coverage is cached next to the input file
  (`.flexpeak_cache/`, or the directory given by `--cache-dir`), so that the
  stages of the run do not re-read the alignments. When `callpeak` finishes,
  successfully or not, the caches of its inputs are deleted, and the default
  `.flexpeak_cache/` directory with them once it is empty. `--keep-cache`
  keeps them, so a repeated call on the same input skips reading it;
  `--no-cache` writes none.

## Call set

A run produces one nested call set, written in four formats:

| File | Format | Contents |
|---|---|---|
| `<name>_peaks.narrowPeak` | ENCODE narrowPeak | sub-peaks and isolated narrow peaks, with summits |
| `<name>_peaks.broadPeak` | ENCODE broadPeak | regions (domains and peaks) |
| `<name>_nested.bed` | BED12 | each region as one record with its sub-peaks as blocks; for genome browsers |
| `<name>_peaks.nested.tsv` | tab-separated | the complete parent–child structure, all statistics, and the full parameter set in the header |

The nested TSV is the native format. It keeps which sub-peaks belong to
which domain, so use it when you rerun `qc` or `stats` later.

Each region carries a fold enrichment, a −log10 p-value and a −log10 q-value
from a negative binomial test on its aggregate counts, using the dispersion of
the sum. Regions of at least 2 kb are labelled `domain` and shorter ones
`peak`; the label is descriptive only. The statistics of a sub-peak inside a
domain are computed relative to the level of that domain.

## Quality-control report

| File | Contents |
|---|---|
| `<name>_qc.txt` | library metrics, a quality grade with its reasons, and call-set statistics |
| `<name>_qc.json` | the same information in machine-readable form, with stage timings and fitted dispersions |
| `<name>_qc.pdf` | with `--figures`: one page with summary values, width and enrichment distributions, calls per chromosome and, where available, the fragment-size distribution and TSS enrichment profile |

`--figures all` instead writes the full set of statistics figures to
`figures/`.

```{figure} figures/tutorial/geo_h3k9me3_chr19_summary.png
:width: 100%
:alt: One-page summary of a call set

Summary page for an H3K9me3 call set on mouse chr19 (GEO GSE285245).
```
