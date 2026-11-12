# FlexPeak

**Flexible, adaptive peak calling for ChIP-seq, CUT&RUN, CUT&Tag and ATAC-seq**

## Description

FlexPeak identifies enriched regions in chromatin profiling data. Narrow peaks
and broad domains are called in a single run and reported as one nested call
set, in which focal sub-peaks are placed inside the domains that contain them.
The type of mark does not have to be declared in advance.

The method combines a negative binomial background model, a three-state hidden
Markov model for segmentation, and region-level significance testing with
false discovery rate control. Parameters are adapted to each sample by a
tuning sweep that validates candidate settings on held-out halves of the
sample's own reads; no pretrained model or training corpus is used. The
package also provides differential peak calling between two conditions,
library quality control, and summary figures.

## Installation

FlexPeak requires Python 3.9 or later. The core depends on NumPy and SciPy
only; file-format support and acceleration are optional extras.

```bash
git clone https://github.com/helenhuangmath/FlexPeak.git
cd FlexPeak
pip install -e ".[bam,bigwig,fast,figures]"
```

| Extra | Installs | Required for |
|---|---|---|
| `bam` | pysam | BAM and CRAM input |
| `bigwig` | pyBigWig | bigWig input |
| `fast` | numba | compiled HMM kernels (a NumPy implementation with matching results is used otherwise) |
| `figures` | matplotlib | `--figures` and `flexpeak stats` |
| `dev` | pytest, ruff | the test suite |

To verify the installation:

```bash
flexpeak --version
python -m pytest tests -q
```

## Quick start

Peak calling on a treatment BAM with an input control, using self-tuning:

```bash
flexpeak callpeak -t chip.bam -c input.bam -n sample --outdir out/
```

Without a control (typical for CUT&Tag and CUT&RUN), the background is
estimated from the treatment with apparent signal excluded:

```bash
flexpeak callpeak -t cutntag.bam -n sample
```

From a bigWig, with a mark preset and no tuning sweep (reproducible to the
bit):

```bash
flexpeak callpeak -t signal.bw --preset h3k27me3 --no-tune -n K27
```

With multiple worker processes, an exclusion list and the QC figure:

```bash
flexpeak callpeak -t chip.bam -c input.bam --threads 8 \
    --blacklist mm10-blacklist.v2.bed.gz --figures -n sample
```

Paired-end ATAC-seq without a control, including ATAC-specific quality
control (TSS enrichment requires a gene annotation):

```bash
flexpeak callpeak -t atac.bam --preset atac --no-tune --threads 8 \
    --blacklist hg38-blacklist.v2.bed.gz --tss gencode.gtf --figures -n sample
```

Each run writes the call set in four formats and a QC report (see
[Output](#output)). The tuning report, which lists the settings evaluated and
the one selected, is printed to standard error and recorded in the output
headers.

## Commands

| Command | Purpose |
|---|---|
| `flexpeak callpeak` | Call nested peaks. Self-tunes by default; `--no-tune` uses the defaults or a `--preset`; `--quick` lets the scan choose the scale and reduces the sweep. |
| `flexpeak scan` | Profile a sample at several bin sizes before a full run: count distributions, clipped background, signal excess, run and gap lengths, and the settings these imply. |
| `flexpeak diff` | Call differential regions between two conditions, with or without replicates. |
| `flexpeak tune` | Run the tuning sweep only and report the objective curve and the selected parameters. |
| `flexpeak qc` | Library and enrichment quality control for a sample, optionally against an existing call set. |
| `flexpeak stats` | Summary statistics and the full set of figures for an existing call set. |

`flexpeak <command> --help` lists every option.

## Input

- **BAM or CRAM**, coordinate-sorted and indexed. Paired-end reads are
  converted to fragments (one per proper pair, template length ≤ 1 kb).
  Single-end reads are extended by a fragment length estimated from strand
  cross-correlation. Duplicate, secondary, supplementary, QC-fail and unmapped
  reads are excluded; the default minimum mapping quality is 30 (`--mapq`).
- **bigWig** signal tracks. These are treated as normalised signal and
  rescaled to pseudo-counts; the resulting p- and q-values are approximate,
  and the output header states this.
- **Chromosomes.** Only assembled chromosomes are analysed by default.
  Unplaced, random and alternate contigs, the mitochondrial genome and EBV
  are excluded. `--chroms` selects a subset and `--all-contigs` includes
  every sequence.
- **Coverage cache.** During a run, binned coverage is cached next to the
  input file (`.flexpeak_cache/`, or the directory given by `--cache-dir`) so
  that the steps of the run do not re-read the alignments. When `callpeak`
  finishes, successfully or not, the caches of its inputs are deleted, and
  the default `.flexpeak_cache/` directory with them once it is empty.
  `--keep-cache` keeps them, so that calling the same input again skips
  reading it; `--no-cache` writes none.

## Output

A run produces one nested call set, written in four formats:

| File | Format | Contents |
|---|---|---|
| `<name>_peaks.narrowPeak` | ENCODE narrowPeak | sub-peaks and isolated narrow peaks, with summits |
| `<name>_peaks.broadPeak` | ENCODE broadPeak | regions (domains and peaks) |
| `<name>_nested.bed` | BED12 | each region as one record with its sub-peaks as blocks; suitable for genome browsers |
| `<name>_peaks.nested.tsv` | tab-separated | the complete parent–child structure, all statistics, and the full parameter set in the header |

and one quality-control report:

| File | Contents |
|---|---|
| `<name>_qc.txt` | library metrics, a quality grade with its reasons, and call-set statistics |
| `<name>_qc.json` | the same information in machine-readable form, with stage timings |
| `<name>_qc.pdf` | written with `--figures`: a single page with summary values, width and enrichment distributions, calls per chromosome and, where available, the fragment-size distribution and the TSS enrichment profile |

`--figures all` writes the full set of statistics figures to `figures/`
instead of the single page.

Each region carries a fold enrichment, −log10 p-value and −log10 q-value from
a negative binomial test on its aggregate counts, using the dispersion of the
sum. Regions of at least 2 kb are labelled `domain` and shorter ones `peak`;
the label is descriptive only. The statistics of a sub-peak inside a domain
are computed relative to the level of that domain.

With `--preset atac`, the report uses metrics appropriate to ATAC-seq: TSS
enrichment (`--tss`), the nucleosome-free and mono-nucleosome fragment
fractions, and FRiP. The normalised strand cross-correlation coefficient is
reported for single-end data only.

## Methods

FlexPeak has five stages and no learned components. A full account, with the
measurements behind each default, is given in [docs/DESIGN.md](docs/DESIGN.md).

1. **Coverage.** Alignments are read once per chromosome into fragment
   coordinates and binned. Chromosomes are processed in parallel.
2. **Background.** A local background is estimated as the maximum over
   several window scales, with apparent signal iteratively excluded so that a
   broad domain does not determine its own background. Counts are modelled
   by a negative binomial distribution with a genome-wide mean–dispersion
   trend.
3. **Segmentation.** A three-state negative binomial hidden Markov model
   (background, weak and strong enrichment) is fitted to coarsened bins with
   per-bin background means. Region boundaries are taken from the posterior
   probability, not from a gap parameter. The segmentation scale and the
   minimum dwell time are derived from the autocorrelation length of the
   sample.
4. **Sub-peaks.** For domain and mixed marks, a focal pass tests sliding
   window sums against the local level; its regions become sub-peaks of the
   domains that contain them and standalone peaks elsewhere. A domain whose
   signal between sub-peaks is not itself enriched is reported as its
   sub-peaks. For narrow marks, sub-peaks are local maxima that persist
   across smoothing bandwidths.
5. **Self-tuning.** Reads on a subset of chromosomes are divided into two
   halves. Each candidate setting is called on one half and scored by how
   well its calls predict the other half, together with the agreement
   between halves and a penalty on called territory. The setting is chosen
   from the plateau of the objective, not from its maximum.

### Quick scan

```bash
flexpeak scan -t chip.bam                       # text report
flexpeak scan -t chip.bam --json -o scan.json   # machine-readable
flexpeak callpeak -t chip.bam -c input.bam -n sample --quick
```

`flexpeak scan` reads six 10 Mb windows from six chromosomes and profiles
them at several bin sizes. Background and candidate signal are separated by
iterative sigma clipping, and the excess over a negative binomial with the
clipped moments is reported as the signal fraction. The sample is classified
as narrow, broad, mixed or null, and the class determines the bin size,
background windows, segmentation scale and merging distance. `callpeak
--quick` applies this recommendation and then tunes a small number of
parameters on four 25 Mb windows. With `--replicates`, the tuning halves are
groups of replicates and replicate support enters the objective. Options
given explicitly on the command line take precedence over the scan.

### Low-depth libraries

For count data, the background is adjusted to sequencing depth; deep
libraries are not affected.

- Each background window is widened until it is expected to contain about
  20 fragments at the fragment rate of the library.
- A stretch is excluded from the background fit only if it exceeds both
  `--background-exclude-fold` times the background and three standard
  deviations of Poisson fragment noise.

To analyse the libraries of one experiment under a common noise model, the
dispersion can be fixed with `--dispersion` and `--segment-dispersion`; the
fitted values of each sample are recorded in `<name>_qc.json`.

### Parallel execution

The background fit and the peak call run with one chromosome per worker
process (`--threads`). The result is identical to that of a serial run. With
`--preset atac`, the expectation–maximisation iterations of the HMM run on a
25 % sample of each chromosome while the posterior is computed on the whole
chromosome (`--hmm-fit-fraction`). Stage timings are printed for every run.

### Differential calling

```bash
flexpeak diff -a KO_rep1.bam KO_rep2.bam -b WT_rep1.bam WT_rep2.bam -n KO_vs_WT
flexpeak diff -a Arm.bw -b Cl13.bw --chroms chr1,chr2 -n Arm_vs_Cl13   # single samples
```

Samples are normalised (by matched background level by default; `--norm tmm`
or `total` are alternatives) and replicates are averaged within each
condition. The proportion of signal from condition A in each bin is modelled
by a three-state HMM (unchanged, higher in A, higher in B) with beta-binomial
emissions. The overdispersion is estimated from replicates where available
and within the EM otherwise. Candidate regions are tested on their aggregate
counts with a region-level overdispersion, followed by Benjamini–Hochberg
correction. A candidate must also be enriched over the control, or over the
local background of the pooled signal, in at least one condition.
`--within-peaks` restricts calls to peaks found in either condition.

Output: `<name>_diffpeaks.narrowPeak` (strand `+` for higher in A, `-` for
higher in B; signal value is the log2 fold change), `<name>_diffpeaks.tsv`
(counts per replicate, posterior, p- and q-values), `<name>_gain.bed` and
`<name>_loss.bed`.

## Parameters

The defaults are intended to be used as they are; the tuning sweep adjusts
the parameters with the largest effect. Each default in `CallParams` is
documented in the source with the measurement on which it is based.

| Option | Default | Description |
|---|---|---|
| `-q`, `--qvalue` | 0.05 | False discovery rate for regions and, separately, for sub-peaks. The region-level correction covers the whole genome, not only the candidates returned. |
| `--min-fold` | 1.2 | Minimum fold enrichment of a region. |
| `--min-width` | 100 bp | Minimum region width. |
| `--max-width` | none (50 kb with `--preset atac`) | Regions wider than this are reported as their sub-peaks. |
| `--bin-size` | 25 bp (preset-dependent) | Reporting resolution. Segmentation uses a coarser scale derived from the data. |
| `--segment-bin-size` | automatic | Segmentation bin size; by default the autocorrelation length divided by 8, with a minimum of 100 bp. |
| `--preset` | none | `tf`, `h3k4me3`, `h3k27ac`, `h3k27me3`, `h3k9me3`, `h3k36me3` or `atac`. Fixed parameter sets for use without tuning. |
| `--no-tune` | off | Skip the tuning sweep. Required for bit-identical repeated runs. |
| `--quick` | off | Use the quick scan to choose the scale and a reduced sweep. |
| `--tune-chroms` | 3 | Number of chromosomes in the tuning subset. |
| `--threads` | all CPUs allocated | Worker processes for reading alignments, fitting the background and calling peaks. |
| `--chroms`, `--all-contigs` | assembled chromosomes | Chromosome subset; `--all-contigs` includes unplaced, random and alternate contigs. |
| `--keep-cache` | off | Keep the coverage cache after the run instead of deleting it. |
| `--blacklist` | none | BED file (optionally gzipped) of regions in which calls are discarded. |
| `--dispersion`, `--segment-dispersion` | fitted per sample | Fix the dispersion trend `a0,a1` per bin or per segmentation bin. |
| `--hmm-fit-fraction` | 1.0 (0.25 with `--preset atac`) | Fraction of each chromosome used for the EM iterations of the HMM. |
| `--tss` | none | GTF, GFF or BED file of transcription start sites; adds TSS enrichment to the QC report. |
| `--figures [qc\|all]` | off | `qc` writes the single-page `<name>_qc.pdf`; `all` writes every statistics figure. |
| `--poisson`, `--background-exclude-fold 0`, `--no-dwell-floor` | off | Ablation options; each disables one component of the method. |

## Python interface

```python
from flexpeak import load_signal, call_peaks, tune, CallParams

treat = load_signal("chip.bam", bin_size=25)        # cached on reuse
ctrl  = load_signal("input.bam", bin_size=25)

result = tune(treat, ctrl, base=CallParams())        # TuneResult: params, curve, report()
print(result.report())

peaks = call_peaks(treat, ctrl, result.params, n_jobs=8)   # NestedPeakSet
peaks.write_all("out/sample")                              # the four call-set files

for region in peaks:
    print(region.chrom, region.start, region.end, region.kind,
          region.fold_enrichment, region.neg_log10_q, len(region.subpeaks))
```

The public classes are `Coverage` (binned signal, with `rebin`, `subset`,
`thin` and `split`), `BackgroundModel`, `NestedPeakSet`, `Region`, `SubPeak`
and `TuningObjective`. `flexpeak.simulate.simulate()` generates narrow, broad
and mixed data with known truth and underlies the tests and the benchmark.

## Benchmark

`Benchmark/` contains a reproducible comparison on simulated data with known
ground truth. MACS2, MACS3, SICER2 and SEACR are run as external programs on
the same alignments and the same control, each at its nominal 5 % false
discovery rate.

```bash
python Benchmark/run_benchmark.py     # writes Benchmark/results/metrics.tsv
python Benchmark/make_figures.py      # writes Benchmark/figures/
```

| Dataset | FlexPeak | MACS2 narrow | MACS2 broad | SICER2 | SEACR (relaxed) |
|---|---:|---:|---:|---:|---:|
| Narrow peaks (TF- or H3K4me3-like) | **91%** | 90% | 80% | 16% | 2% |
| Broad domains (H3K27me3-like) | **100%** | 52% | 94% | 98% | 0% |
| Mixed: domains containing focal peaks (H3K9me3-like) | **99%** | 55% | 94% | 96% | 0% |
| Low quality: 1.6-fold domains, overdispersed | **96%** | 15% | 44% | 63% | 46% |
| Sparse, CUT&RUN-like | **99%** | 20% | 59% | 97% | 94% |
| False calls on a null genome | **2** | 18 | 121 | 430 | 1 |
| Wall time, 4.5 Mb genome including BAM parsing | **3.0 s** | 5.2 s | 5.3 s | 26 s | 3 s |

Values are base-pair F1 against the simulated truth unless stated otherwise
(MACS2 2.2.9.1, MACS3 3.0.4, SICER2 1.0.3, SEACR 1.3; MACS3 matches MACS2 on
every dataset).

![Base-pair F1 by method and dataset](Benchmark/figures/B1_f1_heatmap.png)

**Limitations.** These are simulation results. The simulated background is
drawn from the model family that FlexPeak assumes, which favours it on the
null dataset. MACS2 in narrow mode is slightly more precise on focal peaks,
SICER2 recovers slightly more broad territory, and the self-tuning sweep
reproduces the defaults on these datasets and does not improve on them. The
protocol, the differential comparison and a full discussion are given in
[Benchmark/README.md](Benchmark/README.md). An example run on public
experimental data is provided in
[examples/geo_h3k9me3.py](examples/geo_h3k9me3.py).

## Documentation

The full documentation is at <https://flexpeak.readthedocs.io>, built from
`docs/` with Sphinx (`sphinx-build docs docs/_build/html`).

| Document | Contents |
|---|---|
| [docs/TUTORIAL.md](docs/TUTORIAL.md) | A worked example: a first call, interpreting the output and the QC report, and inspecting a locus. |
| [docs/DESIGN.md](docs/DESIGN.md) | The algorithm, its rationale and the measurements behind each default. |
| [Benchmark/README.md](Benchmark/README.md) | The method comparison, its protocol and threats to validity. |
| [examples/tutorial_simple.py](examples/tutorial_simple.py) | A minimal script using the Python interface. |
| [examples/geo_h3k9me3.py](examples/geo_h3k9me3.py) | Peak calling on public H3K9me3 data from GEO, with checks of the output. |
| [examples/figures.py](examples/figures.py) | Figures assessing correctness on simulated data. |
| [examples/locus_tracks.py](examples/locus_tracks.py) | Locus plots of signal with called regions and sub-peaks. |

## License

FlexPeak is released under the MIT License.

## Package information

| | |
|---|---|
| **Version** | 1.1.0 |
| **Author and maintainer** | Hua Huang |
| **License** | MIT |
| **Language** | Python (≥ 3.9) |
| **Depends** | numpy (≥ 1.21), scipy (≥ 1.7) |
| **Suggests** | pysam, pyBigWig, numba, matplotlib |
| **Topics** | ChIP-seq, CUT&RUN, CUT&Tag, ATAC-seq, peak detection, differential peak calling, epigenetics, quality control |
| **URL** | <https://github.com/helenhuangmath/FlexPeak> |
| **Bug reports** | <https://github.com/helenhuangmath/FlexPeak/issues> |
