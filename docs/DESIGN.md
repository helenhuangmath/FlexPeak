# FlexPeak — Methods

This document describes the statistical method implemented in FlexPeak and
the reasoning behind its main design choices. Usage is described in the
[README](https://github.com/helenhuangmath/FlexPeak#readme) and a worked example in [tutorial](TUTORIAL.md);
the benchmark protocol is in [Benchmark/README.md](https://github.com/helenhuangmath/FlexPeak/blob/main/Benchmark/README.md).
Every default in `flexpeak.caller.CallParams` is documented in the source
with the measurement on which it is based.

## 1. Overview

FlexPeak calls enriched regions in ChIP-seq, CUT&RUN, CUT&Tag and ATAC-seq
data. Its aims are:

- to call narrow peaks and broad domains in one run, reporting focal
  sub-peaks nested inside the domains that contain them;
- to keep the false discovery rate meaningful on overdispersed and shallow
  libraries;
- to require no declared mark type and no hand-tuned parameters.

Every call carries a p-value and a q-value from an explicit statistical
model. The method contains no learned component and uses no training data:
parameters are adapted to each sample at run time from the sample's own
reads (§7).

The pipeline has five stages: coverage (§2), background (§3), segmentation
(§4), region testing (§5) and sub-peak detection (§6), optionally preceded
by self-tuning (§7).

## 2. Signal representation

Alignments are read once per chromosome into fragment coordinates. For
paired-end data each proper pair is one fragment; single-end reads are
extended by a fragment length estimated from strand cross-correlation.
Fragments are binned into per-chromosome coverage arrays at a *reporting*
resolution (10–100 bp depending on the preset). Segmentation (§4) runs at a
coarser, data-derived scale, and its result is mapped back to the reporting
bins, so the reporting bin size sets the resolution of the output, not its
sensitivity.

bigWig input is rescaled to pseudo-counts; p-values computed from it are
flagged as approximate. By default only assembled chromosomes are analysed.

## 3. Background model

### 3.1 Local background

The expected count `mu_i` of bin `i` is a local background in the manner of
MACS: the maximum over several window scales of the local mean. With a
control it is estimated from the control (scaled to the treatment depth);
without one, from the treatment itself.

Estimated from the treatment, a windowed mean contains the signal it is
meant to be compared with, and the fold enrichment of a wide feature
collapses toward 1. The estimate is therefore iterated: bins whose smoothed
signal exceeds `exclude_fold` (default 2) times the current background are
held out and the window means are recomputed from the remaining bins. Two
details make this reliable:

- **Signal is judged on a smoothed track**, at about a tenth of the
  narrowest window. Judging single bins would hold out the upper tail of
  pure noise and bias the background downward.
- **The remaining downward bias of one-sided exclusion is corrected** by the
  median ratio of plain to masked means over the windows from which the
  exclusion removed least. On pure noise this correction restores the plain
  mean.

The background never falls below a floor: the median of the smoothed track,
read at the first smoothing scale at which that median is positive and,
for count data, at which the median window holds at least ten fragments.
The genome mean is not used, because a sample with a large fraction of its
genome in domains carries those domains in its mean.

For domain marks (background windows of at least 500 kb), the exclusion
threshold is additionally anchored on a running median of the smoothed
track, so that widespread moderate enrichment cannot raise its own
background.

### 3.2 Depth-aware background

A window mean is a rate estimate whose precision is set by the number of
fragments in the window, not by its width. For count data of known depth
two rules adapt the background to the library; deep libraries are not
affected by either.

- Each window is widened to the width expected to contain about 20
  fragments at the library's fragment rate.
- A stretch is held out as signal only if it exceeds both `exclude_fold`
  times the background and the background plus three standard deviations of
  Poisson fragment noise for the judged window. Without the second
  condition a single fragment exceeds twice the background on a shallow
  library, every covered stretch is held out, and the background is fitted
  on empty bins.

### 3.3 Dispersion

Counts are modelled as negative binomial,

    count_i ~ NB(mu_i, alpha(mu_i)),   alpha(mu) = a0 + a1 / mu,

with the mean–dispersion trend fitted genome-wide on background bins
(DESeq2/csaw style), so that sparse chromosomes borrow strength from the
whole genome. A Poisson background, which assumes the variance equals the
mean, gives inflated significance on overdispersed libraries: on the
simulated null genome of the benchmark, FlexPeak reports 2 regions with the
negative binomial and 39 with a Poisson background (`--poisson`).

Coverage is autocorrelated, because one fragment spans several bins. The
lag-1 correlation `rho` of background residuals is estimated once per
sample, and the variance of a sum of `n` bins is inflated by the exact
finite-length AR(1) factor

    f(n) = 1 + 2 rho / (1 - rho) * (1 - (1 - rho^n) / (n (1 - rho))).

The trend `(a0, a1)` can be fixed across the libraries of an experiment
(`--dispersion`, `--segment-dispersion`) so that they are tested under a
common noise model.

## 4. Segmentation

A three-state negative binomial hidden Markov model (background, weak and
strong enrichment) is fitted to each chromosome. State `k` emits
`NB(m_k * mu_i, alpha_i)`, so the per-bin background — and with it the
control and local structure — stays inside the model. The multipliers `m_k`
and the transition matrix are estimated by Baum–Welch; the background
multiplier is constrained near 1 and the enriched multipliers are kept
apart. Regions are runs of bins whose posterior probability of enrichment
exceeds `posterior_cutoff`. Boundaries therefore come from the posterior,
not from a fixed gap-merging distance, which is what makes island-merging
callers over-merge.

**Segmentation scale.** Sparse coverage at the reporting resolution is
mostly empty bins; segmented bin by bin, a domain is cut at every run of
zeros. The HMM therefore runs on coarsened bins of `L / 8`, where `L` is the
autocorrelation length of the sample (75th percentile over chromosomes,
since sparsity biases the estimate downward), floored at 100 bp. The
dispersion of a coarsened bin is fitted on the coarsened sums directly; the
per-bin trend evaluated at the summed mean would overstate it by the
coarsening factor and make the states indistinguishable.

**Dwell time.** Unconstrained, Baum–Welch learns enriched runs far shorter
than the features in the data and segments domains into fragments. The
enriched self-transition probability is floored so that the expected dwell
is at least the larger of `min_width` and `L` (capped at `max_dwell`).

On long chromosomes the Baum–Welch iterations can be run on evenly spaced
blocks covering a fraction of the chromosome (`hmm_fit_fraction`; 0.25 in
the ATAC preset), while the posterior is always computed on the whole
chromosome.

## 5. Region testing and FDR

Each region is tested on its aggregate count: the observed sum is compared
with `NB(sum mu_i, alpha_eff)`, where `alpha_eff` is derived from the
variance of the sum, `f(n) * sum(mu_i + alpha_i mu_i^2)` (§3.3), not from
the per-bin dispersion applied to the total. The latter would inflate the
variance in proportion to the region width and penalise broad domains.

A region must also reach a minimum fold enrichment (`min_fold`, default
1.2). This floor is deliberately low: because the background is a maximum
over windows it lies slightly above the true level, and a 1.6-fold domain
measures about 1.35-fold against it. Significance, not the fold floor,
controls false calls.

**FDR scope.** Benjamini–Hochberg correction is applied over the whole
genome, not only over the candidates the HMM returned: the number of tests
is the number of candidate-sized tiles in the genome (genome size divided
by the median candidate width), with untested tiles counted as null. The
HMM screened the genome before returning candidates, and correcting over
the candidates alone would not account for that selection. Sub-peaks are
corrected separately, among themselves.

Nearby regions are joined across a gap of at most `max_gap` only when the
gap is itself enriched to at least `gap_fold_fraction` of the weaker
neighbour's fold. Regions wider than `max_width` (50 kb in the ATAC preset)
are reported as their sub-peaks.

## 6. Sub-peaks and nested output

Sub-peaks are detected in one of two ways.

**Scale-space detection** (narrow marks, and whenever the background
windows are themselves local). The signal within a region is smoothed at a
ladder of bandwidths; local maxima that persist across scales are kept and
single-scale maxima are discarded as noise. Each maximum's footprint is
found by descending from the summit on the first smoothing level. Inside a
domain, a sub-peak is tested against the domain's own level (the median
1 kb-smoothed fold within the domain), since it claims focal structure on
top of the domain. Because candidates are local maxima, their p-values are
not uniform under the null; a pooled outlier gate is applied in addition
to the FDR: the log fold over the domain level must exceed the median of
all candidates by `subpeak_domain_z` (default 4) robust standard
deviations.

**Focal pass** (domain and mixed marks, whose background windows are much
wider than a focal peak). Sliding window sums at one or more focal scales
are tested against a local background: the mean of flanking windows with
the tested window left out, floored at a fraction of the robust background
level. The dispersion of the window sums is measured on the sums
themselves, iteratively excluding windows the fit itself calls enriched.
Significant windows are corrected by BH over all positions and assembled
from fine to coarse scales. Focal regions inside a first-pass region become
its sub-peaks; elsewhere they are standalone peaks.

**Residual test.** A region holding sub-peaks is re-tested with the
sub-peak bins removed. If the remainder is not enriched — or carries less
than `domain_residual_share` of the region's excess — the region is
reported as its sub-peaks rather than as a domain. A promoter cluster is
then a set of peaks, and a Polycomb domain remains a domain.

The output is one nested call set: regions labelled `domain` (≥ 2 kb by
default) or `peak`, each with its sub-peaks. It is written as narrowPeak
(sub-peaks and isolated peaks), broadPeak (regions), BED12 (regions with
sub-peaks as blocks) and a native TSV holding the full structure and the
resolved parameters.

## 7. Self-tuning

The parameters with the largest effect — the posterior cutoff, the fold
floor and the segmentation scale — are chosen per sample without external
data. The reads on a stratified subset of chromosomes are split
binomially into two halves, which are independent draws of the same signal.
Each candidate setting is called on one half and scored by:

1. **Held-out fit.** The fold of each called piece is read from one half,
   and the score is the gain in Poisson log-likelihood of the other half's
   reads under that piecewise enrichment relative to background alone. The
   term is two-sided: missed enrichment forfeits its evidence, called noise
   is not reproduced by the other half, and over-merging dilutes the fold
   over background bins.
2. **Split stability.** Agreement (Jaccard) between the calls on the two
   halves.
3. **Territory penalty.** Stability and reproducibility are both maximised by
   merging everything into a few large regions, so a penalty on called
   territory is required.
4. **Replicate support**, when replicates are given (`--replicates`): the
   tuning halves are then replicate groups, and the share of calls supported
   by the replicates enters the objective.

The setting is taken from the plateau of the objective curve, not from its
maximum. The background is fitted once per half and the HMM once per half
and segmentation setting, since the other parameters only threshold its
output. `--no-tune` with a `--preset` gives fixed, bit-identical runs.

**Quick scan.** `flexpeak scan` (and `callpeak --quick`) profiles six 10 Mb
windows at several bin sizes before the full run. Background and candidate
signal are separated by iterative sigma clipping; the excess of the
observed tail over that of a negative binomial with the clipped moments is
the signal fraction, measured on the raw and on the smoothed track. The
sample is classified as narrow, broad, mixed or null, and the class sets
the bin size, background windows, segmentation scale and merging distance.
`--quick` then tunes a few parameters, one at a time, on four 25 Mb windows.

## 8. Differential calling

`flexpeak diff` compares two conditions. Samples are normalised (by matched
background level by default, or by TMM or library size) and replicates are
averaged within each condition. For each bin, the proportion of signal from
condition A, given the total, is modelled by a three-state HMM (unchanged,
higher in A, higher in B) with beta-binomial emissions, so that depth is
factored out bin by bin and empty bins carry no information. The
overdispersion is estimated from replicate scatter when replicates exist
and within the EM otherwise, with a floor corresponding to a biological
coefficient of variation of 0.2. The dwell floor comes from the
autocorrelation length of the log ratio.

Candidate regions are tested two-sided on their aggregate split with a
region-level overdispersion (replicate scatter of region sums, shrunk
toward the prior), followed by BH correction over the candidates. A
candidate must also be enriched over the control, or over the local
background of the pooled signal, in at least one condition; without this
gate the model follows regional background differences between the
conditions. Bins are coarsened until the two conditions together average
at least ten fragments per bin.

## 9. Quality control

Each run reports library metrics (fragments, fraction of empty bins,
coverage Gini coefficient, autocorrelation length, background dispersion,
fraction of the genome held out of the background), call-set metrics
(numbers of regions, domains and sub-peaks, width distribution and its
bimodality, genome fraction, FRiP) and a coarse grade (A–D) with the
reasons for it. For ATAC-seq the enrichment component is the TSS
enrichment score and the report adds the fragment-size distribution
(nucleosome-free and mono-nucleosome fractions). The normalised strand
cross-correlation coefficient is reported for single-end data only.

## 10. Implementation and performance

- Each chromosome is read once; coverage is built with `bincount` on a
  difference array.
- The background fit, the segmentation and the sub-peak search run with
  one chromosome per worker process; the result is identical to that of a
  serial run.
- On long chromosomes the background is fitted on block means of half the
  exclusion scale and interpolated, since it resolves nothing finer.
- Region statistics come from prefix sums and one vectorised negative
  binomial test per chromosome; sub-peaks are searched only in regions
  that pass the region-level FDR.
- The forward–backward and Viterbi recursions are compiled with numba when
  available, with a NumPy implementation that gives the same results.
- Binned coverage is cached for the duration of a run and deleted at the
  end unless `--keep-cache` is given.

## 11. Evaluation

The benchmark (`Benchmark/`) uses simulated data with known truth: narrow,
broad, mixed, low-quality, sparse and null datasets. Metrics are base-pair
precision, recall, F1 and Jaccard; region-level recall; a fragmentation
index (share of true domains covered by more than one call); a fusion index
(share of calls spanning more than one true domain); boundary error; and the
number of calls on the null genome. Comparators (MACS2, MACS3, SICER2,
SEACR, THOR) are run as external programs on the same alignments and
control, at their nominal FDR, with the exact versions and command lines
recorded.

Each main design choice can be switched off, so its contribution can be
measured:

| Design choice | Switch that disables it |
|---|---|
| Negative binomial background | `--poisson` |
| Signal exclusion in the background | `--background-exclude-fold 0` |
| Depth-aware windows and noise-aware exclusion | `background_depth_aware=False` |
| Dwell-time floor | `--no-dwell-floor` |
| Segmentation-bin floor | `min_segment_bin_bp` |
| Genome-scoped region FDR | `fdr_scope="candidates"` |
| Sub-peaks tested against the domain level | `subpeak_vs_domain=False` |
| Sub-peak boundaries on the smoothed signal | `subpeak_descent_level=0` |
| Held-out term of the tuning objective | `TuningObjective(heldout_weight=0)` |

## 12. Limitations

- The quantitative evaluation is on simulated data whose background is drawn
  from the model family FlexPeak assumes; this favours it on the null
  dataset in particular.
- The self-tuning sweep reproduces the default parameters on the simulated
  datasets and does not yet improve on them; the held-out objective is not
  sensitive enough to boundary precision to choose an absolute
  segmentation scale.
- The background is a maximum over windows and lies somewhat above the true
  level, so calls are conservative; regions are wider than those of
  summit-based callers, with the summit-level detail carried by the
  sub-peaks.
- Without a control, regions of elevated copy number in aneuploid samples
  raise the local background but are not modelled explicitly.
- Gene annotation of peaks and an HTML report are not provided.
