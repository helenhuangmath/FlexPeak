# Parameters and presets

The defaults are meant to be used as they are; the tuning sweep adjusts the
parameters with the largest effect. Each default in
{class}`flexpeak.CallParams` is documented in the source with the measurement
it is based on.

## Main options of `callpeak`

| Option | Default | Description |
|---|---|---|
| `-q`, `--qvalue` | 0.05 | False discovery rate for regions and, separately, for sub-peaks. The region-level correction covers the whole genome, not only the candidates returned. |
| `--min-fold` | 1.2 | Minimum fold enrichment of a region. |
| `--min-width` | 100 bp | Minimum region width. |
| `--max-width` | none (50 kb with `--preset atac`) | Regions wider than this are reported as their sub-peaks. |
| `--bin-size` | 25 bp (preset-dependent) | Reporting resolution. Segmentation uses a coarser scale derived from the data. |
| `--segment-bin-size` | automatic | Segmentation bin size; by default the autocorrelation length divided by 8, with a minimum of 100 bp. |
| `--preset` | none | A fixed parameter set for use without tuning (below). |
| `--no-tune` | off | Skip the tuning sweep. Required for bit-identical repeated runs. |
| `--quick` | off | Use the quick scan to choose the scale, and a reduced sweep. |
| `--tune-chroms` | 3 | Number of chromosomes in the tuning subset. |
| `--threads` | all CPUs allocated | Worker processes for reading alignments, fitting the background and calling peaks. |
| `--chroms`, `--all-contigs` | assembled chromosomes | Chromosome subset; `--all-contigs` includes unplaced, random and alternate contigs. |
| `--keep-cache` | off | Keep the coverage cache after the run. |
| `--blacklist` | none | BED file of regions in which calls are discarded. |
| `--dispersion`, `--segment-dispersion` | fitted per sample | Fix the dispersion trend `a0,a1` per bin or per segmentation bin. |
| `--hmm-fit-fraction` | 1.0 (0.25 with `--preset atac`) | Fraction of each chromosome used for the EM iterations of the HMM. |
| `--tss` | none | GTF, GFF or BED file of transcription start sites; adds TSS enrichment to the QC report. |
| `--figures [qc\|all]` | off | `qc` writes the one-page `<name>_qc.pdf`; `all` writes every statistics figure. |
| `--poisson`, `--background-exclude-fold 0`, `--no-dwell-floor` | off | Ablation options; each disables one component of the method. |

## Presets

Presets are fixed parameter sets, used with `--no-tune` when a run must be
deterministic or the tuning sweep is not wanted.

| Preset | Bin size | Background windows | Typical use |
|---|---|---|---|
| `tf` | 10 bp | 1, 5, 10 kb | transcription factors, focal CUT&RUN |
| `h3k4me3` | 25 bp | 1, 5, 10 kb | promoter marks |
| `h3k27ac` | 25 bp | 10, 50 kb | enhancer and promoter acetylation |
| `h3k27me3` | 100 bp | 1, 5 Mb | Polycomb domains |
| `h3k9me3` | 100 bp | 1, 5 Mb | heterochromatin, domains with focal peaks |
| `h3k36me3` | 100 bp | 1, 5 Mb | gene-body marks |
| `atac` | 10 bp | 1, 5, 10 kb | ATAC-seq; 50 kb width cap, sampled HMM fit |

## Low-depth libraries

For count data, the background adapts to sequencing depth. Deep libraries are
not affected.

- Each background window is widened until it is expected to contain about 20
  fragments at the library's fragment rate.
- A stretch is excluded from the background fit only if it exceeds both
  `--background-exclude-fold` times the background and three standard
  deviations of Poisson fragment noise.

To analyse all libraries of one experiment under a common noise model, fix
the dispersion with `--dispersion` and `--segment-dispersion`. The fitted
values of each sample are recorded in `<name>_qc.json`.

## Parallel execution

The background fit and the peak call run with one chromosome per worker
process (`--threads`). The result is identical to that of a serial run.
