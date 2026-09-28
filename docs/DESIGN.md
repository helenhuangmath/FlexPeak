# FlexPeak — Design & Development Plan

Flexible, adaptive peak calling for ChIP-seq / CUT&RUN / CUT&Tag / ATAC-seq.

Status: design draft, v0. Nothing implemented yet.

**Scope decision (recorded):** FlexPeak has **no pretrained model, no training corpus, and no ML dependency** on any path. Adaptivity comes from per-sample self-tuning at runtime (§8). See §13 for what was considered and dropped, and why.

---

## 1. Problem statement

Pain points restated as testable engineering targets:

| Pain point | Concrete target |
|---|---|
| MACS2 handles narrow well, broad poorly | On H3K27me3 / H3K9me3 / H3K36me3, recover ≥90% of deep-coverage domain territory at matched FDR |
| SICER/epic2 over-merge | Median domain boundary error < 1 kb vs. deep-data reference; fusion index (§9.4) below MACS2-broad and epic2 |
| Mixed marks are unrepresented | Emit **nested** calls: broad domain + focal sub-peaks inside it, in one run, from one model |
| Low-quality data degrades badly | Given a 10M-read subsample of a 150M-read library, recover ≥70% of the deep-data peak set (MACS2 typically ~40–55%) |
| Parameters must be hand-tuned per mark | Zero required parameters; auto-tune from the data in <2 min |

Everything below serves those five numbers. If a feature doesn't move one, it's v2.

---

## 2. Positioning and prior art

### The differentiator

**Nested multi-scale output + corpus-free adaptive parameterization + graceful degradation on shallow data.** Not "we used AI" — lead with the biology and with the fact that nothing was pretrained.

### Genuinely underserved

Mixed-scale marks. H3K9me3 has megabase domains *and* focal ZNF-cluster peaks; H3K4me3 has broad domains at cell-identity genes; H3K36me3 tracks gene bodies with internal structure. Every existing tool forces a choice of scale and a second run. This is the strongest novelty claim.

### Prior art to know, cite, and benchmark against

| Tool | Relevance |
|---|---|
| **HMMRATAC** | HMM over multi-scale signal, explicitly models mixed peak structure (ATAC). Closest conceptual precedent — differentiate carefully |
| **LanceOtron** | CNN peak caller; already occupies the "ML peak calling" position. FlexPeak's corpus-free design is the contrast |
| **csaw / MOSAiCS** | NB windowed testing; GLM background with mappability/GC covariates. Build on this, don't reinvent |
| **SEACR** | Standard for sparse CUT&RUN with no control; auto-selects its own threshold. Direct competitor on in-house data |
| **MACS3** | The baseline that must be matched on narrow marks before anything else matters |
| **epic2 / SICER2** | The broad-mark baseline; the over-merge behaviour FlexPeak claims to fix |
| **Genrich, HOMER, PePr, JAMM** | Secondary comparators; JAMM is multiscale |

---

## 3. Algorithmic core

Design principle: **an auditable statistical caller end to end.** Every call carries a p/q-value a reviewer can reason about. No learned component sits between the data and the output.

### 3.1 Signal representation

- Read/fragment coverage binned at adaptive resolution (default 25 bp; 10 bp for TF/ATAC, 200 bp for broad marks — chosen by the tuner).
- Paired-end: use actual fragments. Single-end: estimate fragment length by strand cross-correlation, extend.
- Store as per-chromosome `int32`/`float32` numpy arrays. Cache to disk (npz/zarr) so re-runs skip BAM parsing entirely. This is what makes the §8 sweep affordable.

### 3.2 Background model — negative binomial GLM with covariates

MACS2's local Poisson λ is the root cause of its poor low-SNR behaviour: Poisson assumes mean = variance, real ChIP background is overdispersed, so noisy/shallow libraries produce inflated significance and a meaningless FDR.

```
count_i ~ NB(mu_i, phi(mu_i))
log(mu_i) = b0 + b1*log(control_i + c) + b2*mappability_i + s(GC_i) + b3*log(local_lambda_i) + offset(depth)
```

- Dispersion `phi` from a genome-wide mean–variance trend fit on background bins (DESeq2/csaw-style), so low-count bins borrow strength.
- Covariates: mappability, GC, blacklist, optionally copy number (essential for aneuploid samples — MACS2 ignores this and produces amplification artifacts).
- **No-control mode is first-class:** local background from the treatment itself plus covariates. Most CUT&Tag, and much real-world data, has no matched input.

This is the highest-value component in the plan and involves no ML at all. Its specific testable consequence is p-value calibration on negative controls (§9.4).

### 3.3 Multi-scale segmentation — the mixed-peak engine

Two coupled layers, one pass:

**Layer 1 — domain segmentation (coarse).** NB-HMM over coarse bins, states `{background, weak-enriched, strong-enriched}`. Forward–backward gives per-bin posterior enrichment; Viterbi gives boundaries. Boundaries are *inferred*, not set by a `--gap-size` guess — which is precisely why SICER over-merges.

**Layer 2 — sub-peak detection (fine).** Scale-space / à trous wavelet decomposition of the fine-binned signal. Smooth at a bandwidth ladder (50, 150, 500, 1500, 5000 bp), find local maxima at each scale, link maxima across scales into a scale-space tree. A feature persisting across many scales is real structure; one appearing at a single scale is noise. Standard practice in astronomical source detection, and the right tool for "sharp and broad in the same track."

**Merge.** Sub-peaks are assigned to their containing domain:

```
domain_00123   chr1  1,200,000  1,460,000  q=1e-31  width=260kb  n_subpeaks=4
  └ peak_00123.1  chr1  1,212,400  1,213,900  summit=1,213,050  q=1e-18
  └ peak_00123.2  ...
```

Emit `narrowPeak` (sub-peaks), `broadPeak`/`gappedPeak` (domains), plus a native nested format (BED12 or Parquet) preserving the parent/child link. Isolated sub-peaks with no enclosing domain are ordinary narrow peaks. **Nothing requires the user to declare "narrow" or "broad" up front** — that is the whole point.

### 3.4 The tuning objective — making "looks right in IGV" explicit

Automating parameter selection is not a search problem (grid search is trivial); it is an *objective* problem. Expert tuning by browser inspection optimizes something real but unstated: peaks land where signal visibly is, boundaries hug the enrichment, no fragmentation, no fusion. Automation requires naming that. Four objective terms, none requiring manual inspection:

1. **Replicate reproducibility.** How many peaks are reproducible across true replicates (IDR, or overlap consistency). The closest automatic proxy to visual correctness: fragmenting a domain and calling noise both reproduce poorly.
2. **Subsampling stability.** Calls surviving a 50% read subsample are not fitting noise.
3. **Motif central enrichment (TF mode).** A real CTCF peak carries the motif near its summit. A precision signal requiring no reference peak set. **Use CTCF as the calibration anchor for the whole project.**
4. **FRiP + signal partition,** constrained by per-mark width and count priors.

Terms 1 and 2 are **maximized by over-merging** — one giant peak per chromosome is perfectly stable and perfectly reproducible. A **territory/width penalty is therefore mandatory**, and at least one ungameable orthogonal term (3 or 4) must carry weight. See §12.

Select the **plateau of the objective curve, not the argmax** — argmax overfits the metric exactly as eyeballing overfits one locus.

**Legacy hand-tuned parameters as a validation set.** Existing lab MACS2 wrappers (mm10 CUT&RUN, `Code/legacy/`) encode expert choices across the narrow→broad spectrum:

| Mark | `-p` | `min-length` | `max-gap` | `slocal` | `llocal` | post-filter |
|---|---|---|---|---|---|---|
| CTCF | 1e-10 | default | default | 1000 | 10000 | — |
| H3K27ac | 1e-10 | default | default | 3000 | 30000 | — |
| H3K27me3 | 1e-8 | 500 | 800 | 5000 | 50000 | FC>1.1 |
| H3K9me3 | 1e-8 | 300 | 800 | 5000 | 50000 | FC>1.2 |

Monotone in mark breadth across every parameter. **Validation question: does the self-tuner independently recover these settings without being shown them?** Agreement with expert choice is a strong paper figure and the labels already exist.

---

## 4. Package architecture

```
flexpeak/
├── io/            bam.py (pysam), bigwig.py (pyBigWig), bed.py, cache.py
├── signal/        coverage.py, fragment.py, normalize.py, scalespace.py
├── background/    nbglm.py, dispersion.py, covariates.py (mappability/GC/CN)
├── segment/       hmm.py, subpeak.py, merge.py, nested.py
├── stats/         nbtest.py, fdr.py, idr.py
├── tune/          objective.py, sweep.py, subsample.py, presets.py
├── qc/            metrics.py, report.py
├── plots/         style.py (Arial, vector PDF), stats.py (per-run statistics figures)
├── annotate/      genes.py, features.py, motifs.py, repeats.py
├── bench/         harness.py, truth.py, metrics.py, competitors.py
├── cli.py
└── data/          blacklists, chrom sizes, mark presets
```

**CLI** — deliberately MACS2-shaped so people can switch in one line:

```bash
flexpeak callpeak -t chip.bam -c input.bam -g hs -n H3K9me3_sample   # self-tunes, nested output
flexpeak callpeak -t sig.bw --mark H3K27me3                          # bigWig input
flexpeak tune     -t chip.bam --chroms chr1,chr19 --report           # inspect the objective curve
flexpeak qc       -t chip.bam -p peaks.nested.bed -o qc.html
flexpeak stats    -p peaks.nested.tsv --outdir figures              # width/count/enrichment figures, PDF
flexpeak annotate -p peaks.nested.bed --gtf gencode.v44.gtf
flexpeak callpeak ... --preset h3k27me3 --no-tune                    # skip the sweep, use a preset
```

**Dependencies:** numpy, scipy, pysam, pyBigWig, pandas, **pyranges** or **bioframe** (interval ops — do not write your own), numba (hot loops), jinja2 + plotly (report). No scikit-learn, no LightGBM, no torch. Install is `pip install flexpeak`, full stop — no model download, no cache dir, no checksums, no version pinning of weights.

---

### 3.2b Signal exclusion — the background must not contain the peaks

A windowed mean over the treatment includes whatever it is about to be divided
by. A feature filling an appreciable fraction of its window raises its own
background and its fold enrichment collapses toward 1 — worse the wider the
feature, which is exactly backwards for a broad-mark caller. Measured on the
CUT&RUN samples in `testing/`, `mu` inside an H3K27ac peak zone sat at ~1000
against a genome mean of 10, and H3K27me3 domains were shredded into ~1 kb
fragments.

The fit therefore iterates: estimate, hold out bins that look enriched, re-estimate
from the rest. Two details are load-bearing:

- **Judge the smoothed signal, not single bins.** At realistic dispersion a large
  slice of pure noise sits above 2× its own mean, so a per-bin mask eats the
  background's upper tail. Averaging over ~a tenth of the narrowest window shrinks
  noise by √n while leaving a real domain at its true fold. Without this the null
  false-positive territory went from 1.3% to 12.9% of the genome.
- **Correct the residual trimming bias.** One-sided exclusion still biases the
  mean down. The windows where exclusion removed least are the windows with least
  signal, so `median(plain/masked)` over those measures the bias and nothing else.
  On pure noise every window is such a window and the estimator collapses back to
  the plain mean.

Ablate with `--background-exclude-fold 0`.

### 3.3b Dwell time — the HMM must not shred what it segments

Baum-Welch re-estimates the transition matrix, and on real coverage it reliably
collapses the enriched dwell time: measured self-transitions of 0.60–0.86, i.e.
mean enriched runs of 63–720 bp, on marks whose own autocorrelation length is
0.9–7.4 kb. Inside a real domain individual bins fluctuate below the enriched
mean, and a free transition matrix buys likelihood by flickering rather than
staying put.

The enriched self-transition is therefore floored at the larger of `min_width`
and the signal's measured autocorrelation length — the mark's own scale, read off
the sample rather than declared. The model may still learn longer dwells; it may
not spend probability mass on runs shorter than the features being called.
Ablate with `--no-dwell-floor`.

### 3.2c The background floor must be the background, not the genome mean

The local background is floored at a genome-wide average — the guard that stops
peaks being called in anomalously quiet regions (MACS's `lambda_bg`). Using the
*genome mean* for it is wrong as soon as an appreciable fraction of the genome is
enriched: a sample with a fifth of the genome in H3K9me3/H3K27me3 domains carries
those domains in its own mean, so the floor sits above the real background and the
weaker half of every domain scores as depleted. Signal exclusion already
identifies background bins, so the floor is re-read from them.

Effect on H3K27me3 (chr4, default settings): 1,363 regions of median 7.1 kb
covering 10.2% → **765 regions of median 26.7 kb covering 17.5%**, FRiP 0.883 →
0.926, against a signal envelope covering 20.2%. Null false-positive territory
went *down* (1.26% → 1.12% at dispersion 0.5).

### 3.3c Segmentation scale — segment coarse, report fine

Per-bin state inference needs a bin to carry a decision. These libraries are
80–95% empty at their reporting bin size, so inside a real domain the HMM meets
long runs of zeros, reads them as background, and cuts there: H3K27me3 came back
as 75,069 regions of median 1.7 kb instead of domains.

The HMM therefore segments on coarsened bins and the posterior is mapped back to
reporting resolution. Only the segmentation is coarsened — region statistics, the
region-level test and sub-peak detection all still run on the fine bins, so the
only thing given up is boundary quantisation.

The scale is `signal_scale_bp / 8`, where `signal_scale_bp` is the autocorrelation
length taken at the 75th percentile across chromosomes (the estimator is biased
*downward* by sparsity, so low readings are coverage artefacts — on H3K9me3 the
per-chromosome estimates spanned 12x). Choosing it from *emptiness* instead was
tried and is exactly backwards: H3K27ac is the emptiest of the three test samples
(96.7% of 25 bp bins) and is the one that must not be coarsened.

Swept on the three CUT&RUN samples, `scale / 8` lands on the best or near-best
setting for each by signal-per-called-base. It leaves H3K9me3 essentially
uncoarsened, which is correct for it: that sample's signal is punctate, and
forcing it into domains cost FRiP 0.45 → 0.21. Override with `--segment-bin-size`.

One thing to get right when coarsening: **the dispersion of a sum of bins is not
the per-bin trend evaluated at the summed mean.** For homogeneous bins it is
`alpha/k`, so reading the trend at `k*mu` leaves it a factor of k too large and
the states become indistinguishable exactly when the bins get coarse enough to be
useful. The symptom is diagnostic: called territory *falls* as segmentation
coarsens (H3K9me3, 5.7% → 0.8%) when it should rise.

Two things that did *not* work, kept as ablations rather than deleted so the
measurement survives:

- **State-dependent dispersion** (`hmm_fit_state_dispersion`). The emission model
  does under-disperse inside regions — observed variance 50,098 against a model
  19,164 on H3K9me3 — but widening it is less discriminating *everywhere*: called
  territory fell 4.07% → 0.85% while median width barely moved. Fragmentation is
  held together by the transition prior, not by the emissions.
- **A per-chromosome dwell scale.** Unstable enough on sparse chromosomes to
  switch the dwell floor off on five of seven; now computed once per sample.

## 5. QC module

Computed on every run, emitted as one self-contained HTML report:

- **Library:** depth, duplicate rate, NRF/PBC1/PBC2, fragment size distribution, chrM fraction.
- **Enrichment:** FRiP, NSC/RSC, fingerprint/JSD vs. control, TSS enrichment, background dispersion.
- **Peaks:** count, width distribution (bimodality → mixed-mark detection), score distribution, blacklist overlap %, saturation curve from subsampling (answers "do I need to sequence deeper?" — genuinely useful, rarely provided).
- **Reproducibility:** IDR / self-consistency when replicates are given.
- **Tuning transparency:** the objective curve, the selected plateau, the resolved parameter set, and any instability across sampled chromosomes.
- **A quality grade (A–D)** tied to which parameters the tuner adapted and why. QC as an *explanation of the caller's behaviour* rather than a wall of numbers is a real differentiator.

## 6. Annotation module

Nearest gene/TSS with distance and strand; genomic feature distribution (promoter/5'UTR/exon/intron/3'UTR/intergenic); CpG islands; **repeat class overlap** (mandatory for H3K9me3 interpretation — most tools skip it); cCRE/enhancer-catalog overlap; motif enrichment in TF mode; optional GO enrichment of nearest genes. Ship hg38/hg19/mm10/mm39, extensible via GTF.

## 7. Speed

Target: within 1.5× of MACS2 wall-clock on the same BAM, using more cores.

1. **Bin, don't iterate per-bp.** Base-pair resolution only inside candidate regions for summit refinement. Biggest single win.
2. **Per-chromosome multiprocessing** — embarrassingly parallel; MACS2 is single-threaded. This is where you actually beat it.
3. **Numba** for HMM forward–backward and region linking. Rust/pyo3 only if profiling demands it — the maintenance cost is real.
4. **Cache the binned coverage.** Re-runs and every sweep point skip BAM I/O entirely.
5. **Stratified chromosome subset for tuning:** one large gene-dense (chr1), one gene-poor (chr4/chr18), one small (chr19). One chromosome is not representative for broad marks and chr1 alone biases gene-dense. Exclude chrY/chrM. Auto-escalate when estimates disagree across the subset, and say so in the report.
6. Accept CRAM; support bigWig input to skip alignment parsing entirely.

---

## 8. Self-tuning at runtime — no training corpus

The tuner learns from the sample in front of it. This removes the largest cost, the largest risk, and the largest reviewer objection from the original plan simultaneously.

### 8.1 The insight

Depth-subsampling simulation does not require a public corpus. It requires *a library deeper than the one being called on* — and every sample is deeper than a subsample of itself.

1. Load the treatment BAM, build binned coverage on the stratified chromosome subset (§7.5), cache it.
2. Create degraded copies at 50% and 25% depth, two seeds each — resample the cached fragment set, no re-parsing.
3. Sweep the parameter grid (~100 points). For each setting, call on full depth and on the degraded copies.
4. Score with the §3.4 objective.
5. Take the plateau. Report the choice and the curve in QC.

Coverage is parsed once; each grid point is segmentation over in-memory arrays. A 100-point sweep on two chromosomes is seconds to low minutes, inside the §7 budget.

### 8.2 Why this is better, not merely cheaper

- **No training-set bias** to defend — no cell-type imbalance, no mark imbalance, no ChIP→CUT&RUN domain shift, no "does it generalize to my organism."
- **Works out of distribution by construction:** novel marks, non-model organisms, new assay chemistries, unusual protocols.
- **No model versioning, download, or optional extras.**
- **Fully auditable:** the decision is a scored curve over an explicit grid, not a learned function.
- **"No training data required" is a selling point** against LanceOtron and similar, not a weakness.

### 8.3 Failure modes to guard

- **Over-merge drift** (§3.4) — the objective's built-in bias. Mandatory territory penalty; validate specifically on broad marks, where the pull is strongest.
- **Unrepresentative chromosome subset** — stratify, escalate on disagreement, surface instability in QC.
- **Sweep cost on very large libraries** — cap via subset size, not by dropping grid points silently; log any reduction.

### 8.4 Presets as the fallback

Ship per-mark/per-assay presets derived from the benchmark runs. `--no-tune --preset X` must always be available for pipelines that need bit-identical reruns, and is the fallback whenever the objective curve has no clear plateau.

---

## 9. Evaluation plan

Build the harness in **P0, before the caller**. A benchmark you can re-run on every commit is the difference between a tool and a paper.

### 9.1 Principles

- **Freeze the benchmark before optimizing against it.** Reserve a held-out set of datasets never examined during development; report on it once, at the end.
- **Report per-dataset, never only averages.** Aggregate numbers hide the mark where the method fails.
- **Every design claim maps to one figure and one ablation.** A claim with no ablation is marketing.
- **No single ground truth exists.** Use four independent reference types and be explicit about what each can and cannot support.

### 9.2 Reference construction

| ID | Reference | Built from | Supports | Does **not** support |
|---|---|---|---|---|
| **R1** | Deep-library reference | Full-depth calls, consensus of ≥3 callers ∩ IDR reproducibility | Recall, territory, boundary accuracy, degradation curves | Precision — it is caller-derived and inherits caller bias |
| **R2** | Motif-based (TF) | Central motif enrichment at summits | Precision, summit accuracy | Histone marks |
| **R3** | Orthogonal assay concordance | ATAC/DNase, RNA-seq (H3K36me3/H3K4me3), ChromHMM states, cCREs, repeat classes (H3K9me3) | Biological plausibility, precision-like evidence | Exact boundaries |
| **R4** | Negative controls | Input-vs-input, IgG-vs-IgG, replicate-vs-replicate, circularly-shifted signal | False positive rate, p-value calibration | Recall |

**R1 is used for recall and ranking only.** Reporting precision against a MACS2-derived reference would measure agreement with MACS2, not correctness — state this explicitly in the paper before a reviewer does.

**R4 is the cheapest and most convincing test in the whole plan** and is usually omitted by peak-caller papers: calling input against input should yield ≈0 peaks, and the p-value distribution should be uniform. This directly tests the §3.2 NB claim.

### 9.3 Datasets

| Class | Content | Purpose |
|---|---|---|
| Narrow | CTCF, 2–3 other TFs, H3K4me3 (deep, replicated, with controls) | Parity gate vs. MACS3 |
| Broad | H3K27me3, H3K9me3, H3K36me3 | Domain accuracy, anti-over-merge |
| Mixed | H3K9me3 (domains + ZNF focal), H3K4me3 broad domains | Nested-output claim |
| Assay transfer | Public CUT&RUN / CUT&Tag (Henikoff, 4DN) | Generalization beyond ChIP |
| In-house | mm10 CUT&RUN d30, replicated, 4 marks | Real-use validation + §3.4 expert-parameter check |
| Species | ≥2 mouse datasets | Non-human sanity |
| Held-out | 3–4 datasets sequestered until final evaluation | Benchmark-overfitting control |

~15–20 experiments total, **evaluation only** — no sweep corpus, no feature extraction, no label pipeline.

### 9.4 Metrics

**Agreement with reference (R1)**
- Base-pair level: precision / recall / F1, Jaccard on territory.
- Peak level: match at ≥50% reciprocal overlap → recall, precision.
- **Threshold-free:** AUPRC from score ranking. Also compare at *matched call count* and at *matched FDR* — otherwise you are comparing thresholds, not methods.

**Boundary quality**
- Median and IQR of |Δstart|, |Δend| against R1 domains.
- **Fragmentation index:** fraction of reference domains covered by >1 call. Tests the MACS2-broad weakness.
- **Fusion index:** fraction of calls spanning >1 reference domain. **Directly tests the SICER/epic2 over-merge claim, and equally guards against FlexPeak's own over-merge drift (§8.3).**

**Nested output** (no competitor produces this — evaluate against R1 nested reference)
- Sub-peak recall within correctly-called domains; summit displacement; false sub-peak rate inside flat domains.

**Precision without a peak set**
- R2: fraction of calls with centrally-enriched motif; motif-score AUC vs. rank.
- R3: enrichment of calls in expected chromatin states / repeat classes; correlation with expression for H3K36me3.

**Calibration and specificity (R4)**
- Peaks called on input-vs-input at nominal FDR 0.05 (target: ≈0).
- QQ plot of p-values under the null; deviation from uniform. **This is the single cleanest demonstration that NB beats Poisson.**

**Robustness**
- Degradation curves: F1 vs. read depth (30M → 2M). **Headline figure.**
- Area under the degradation curve as a single summary number.
- Variance across subsample seeds.

**Resources**
- Wall-clock, peak RSS, thread scaling (1/4/8/16), on identical hardware and input.

### 9.5 Ablations — one per design claim

| Claim | Ablation | Expected effect |
|---|---|---|
| NB background beats Poisson | Swap NB → Poisson | Calibration degrades (R4 QQ), low-depth F1 drops |
| Covariates matter | Drop mappability/GC/CN | FP rate rises in low-mappability and amplified regions |
| HMM beats gap-linking | Swap HMM → fixed-gap merge | Fusion and fragmentation indices rise |
| Multi-scale enables mixed calls | Single-scale only | Sub-peak recall collapses on H3K9me3 |
| Self-tuning adds value | Tuned vs. defaults vs. expert-tuned | Tuned ≈ expert-tuned ≫ defaults |
| Plateau selection beats argmax | Select argmax instead | Higher variance across subsample seeds |

### 9.6 Statistical reporting

Bootstrap CIs over peaks/regions; fixed seeds recorded; per-dataset tables in supplement; harness re-runnable via one command and executed in CI on a small fixture set to catch regressions.

---

## 10. Method comparison plan

### 10.1 Comparators

| Tool | Mode(s) | Why included |
|---|---|---|
| **MACS3** | narrow; broad | Primary baseline; parity gate |
| **MACS2** | expert-tuned (`Code/legacy/`) | The realistic bar — what a careful user actually achieves |
| **epic2 / SICER2** | default; tuned | Broad-mark baseline; over-merge claim |
| **SEACR** | stringent; relaxed | CUT&RUN baseline, no-control setting |
| **HMMRATAC** | default | Closest conceptual precedent; must be addressed directly |
| **LanceOtron** | default | ML-caller contrast |
| **Genrich, HOMER, JAMM** | default | Secondary breadth |

### 10.2 Fairness protocol — non-negotiable

Comparing self-tuned FlexPeak against default-parameter competitors is a strawman, and reviewers will catch it. Every competitor is run in **three configurations**, all reported:

1. **Tool defaults.**
2. **Author-recommended settings** for that mark, from the tool's own docs/paper.
3. **Expert-tuned** — for MACS2, the `Code/legacy/` parameters; for others, a comparable good-faith sweep using the same objective FlexPeak uses.

Additional controls:
- Identical inputs: same BAMs, same dedup, same blacklist, same effective genome size.
- Compare at **matched FDR** *and* **matched call count**; report both.
- Same hardware, same thread count for timing.
- Record exact version and full command line for every tool run; commit them to the repo.
- Where a tool cannot produce an output type (e.g. nested calls), say so plainly rather than scoring it zero.

### 10.3 Claims → evidence map

| Claim | Primary figure | Comparators | Reference |
|---|---|---|---|
| Matches MACS on narrow marks | F1 + AUPRC, narrow panel | MACS3, MACS2-tuned | R1, R2 |
| Better on broad domains | Territory F1 + boundary error | epic2, MACS2-broad | R1 |
| Does not over-merge | Fusion index | epic2, SICER2 | R1 |
| Handles mixed marks | Sub-peak recall in domains, H3K9me3 browser panel | none produce nested — show qualitative + quantitative | R1, R3 |
| Robust on low-quality data | **Degradation curve (headline)** | MACS3, epic2, SEACR | R1 |
| Correct FDR | QQ plot + input-vs-input peak count | MACS3, epic2 | R4 |
| Adaptive without training | Tuned vs. expert-tuned parameter agreement | MACS2-tuned | §3.4 table |
| Fast | Wall-clock + thread scaling | all | — |

### 10.4 Guarding against benchmark overfitting

Development uses the open set only. The held-out datasets (§9.3) are run **once**, at the end, and reported whether or not they flatter the method. If held-out performance diverges from development performance, report the gap — that finding is more valuable than a clean win.

---

## 11. Phased roadmap

| Phase | Deliverable | Effort |
|---|---|---|
| **P0** | Scaffolding, packaging, CI, `NestedPeakSet` contract, BAM/bigWig I/O, binned coverage + cache, **evaluation harness + reference construction (§9) + competitor wrappers (§10)** | 4–5 weeks |
| **P1** | NB-GLM background, NB testing, FDR, narrow calling. **Gate: match MACS3 on narrow marks; pass R4 calibration** | 3–4 weeks |
| **P2** | NB-HMM domains + scale-space sub-peaks + nested output. **Gate: beat epic2 on fusion index; demonstrate nested calls on H3K9me3** | 5–7 weeks |
| **P3** | Self-tuning: subsample simulation, objective, sweep, plateau selection, presets. **Gate: recovers §3.4 expert parameters unseen** | 2–3 weeks |
| **P4** | QC + HTML report + annotation. **Gate: the degradation-curve figure** | 3–4 weeks |
| **P5** | Held-out evaluation, docs, bioconda/PyPI, tutorials, preprint | 4–5 weeks |

P0–P2 is already a publishable, useful tool. P3 adds adaptivity with no training data and no ML dependency.

## 12. Risks

| Risk | Mitigation |
|---|---|
| Self-tuning objective gamed by over-merging | Mandatory territory penalty + ungameable orthogonal term (§3.4); fusion index tracked as a gate, not just a metric |
| Chromosome subset unrepresentative | Stratified subset; auto-escalate on disagreement; surface instability in QC |
| Sweep too slow on large libraries | Coverage parsed once; grid points are array ops; cap subset size and log any reduction |
| Benchmark overfitting | Held-out datasets, run once (§10.4) |
| Prior art overlap (HMMRATAC, LanceOtron) | Differentiate on nested output + corpus-free adaptivity; benchmark both directly |
| Scope creep (differential binding, motif discovery, super-enhancers) | Out of scope for v1 |
| Name collision | Check PyPI/bioconda/Scholar for "FlexPeak" before announcing |
| Reproducibility complaints | Fixed seeds; full resolved parameter set in every output header; `--no-tune --preset` for bit-identical reruns |

## 13. Explicitly out of scope (considered and dropped)

Recorded so the reasoning isn't relitigated later.

- **ENCODE-scale training corpus and pretrained tuner.** Dropped: per-sample self-tuning (§8) achieves the same goal with no corpus, no domain-shift problem, and no model distribution. Several thousand CPU-hours and ~6–9 weeks saved. ENCODE data is still used, for **evaluation only** (§9.3).
- **Training on ENCODE peak files.** Circular — ENCODE peaks are MACS2 output, so the ceiling would be MACS2. This reasoning still governs R1's restricted role (§9.2).
- **Learned candidate rescoring (shape priors).** Genuinely needs cross-sample data, so it cannot be done in-sample. Post-v1 at the earliest, and only if P1–P3 leave a measurable gap on the degradation curve.
- **AlphaGenome / sequence-model priors.** API-gated and non-redistributable, so unusable in a bioconda package; and using predicted signal as truth imports reference bias that erases the cell-type-specific peaks that matter most. If ever revisited, open weights only (Borzoi), optional, off by default.
- **Differential binding, motif discovery, super-enhancer stitching.** Peak calling + QC + annotation only for v1.

## 14. Immediate next steps

1. Check the `FlexPeak` name on PyPI, bioconda, and Scholar.
2. Pick a license (MIT or BSD-3 — GPL blocks some institutional users).
3. Scaffold: `pyproject.toml`, `src/` layout, ruff + pytest + GitHub Actions, Python ≥3.10 (local default is 3.9 — set up a dedicated env).
4. Define the `NestedPeakSet` contract — cheapest to change now, most expensive later.
5. Assemble the ~15–20 evaluation datasets (§9.3) and sequester the held-out set immediately, before any development.
6. Build the evaluation harness and competitor wrappers (§10.1) **before** the caller.
7. Build P1 and prove parity with MACS3 on CTCF, plus R4 calibration, before anything adaptive.
