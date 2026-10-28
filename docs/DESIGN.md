# FlexPeak — Design & Development Plan

Flexible, adaptive peak calling for ChIP-seq / CUT&RUN / CUT&Tag / ATAC-seq.

Status: v0.1 — §3, §7 and §8 are implemented and measured; §5 (QC) is partial; §6 (annotation) is not started. Sections written as plans before implementation have been revised to record what was built and what was measured; where a measurement overturned the plan, the plan is kept and the correction follows it, so the reasoning survives. The simulated-data comparison that produced the numbers quoted here is `Benchmark/README.md`; experimental-data validation (§9–10) is still ahead.

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

- Read/fragment coverage binned at a *reporting* resolution (default 25 bp; 10 bp for TF/ATAC presets, 100 bp for broad-mark presets). The HMM segments at a coarser, data-derived scale (§3.3c) and the posterior is mapped back, so the reporting bin is a resolution choice, not a sensitivity choice.
- Paired-end: use actual fragments. Single-end: estimate fragment length by strand cross-correlation, extend.
- The BAM is scanned **once** per chromosome into integer arrays of fragment anchors (forward starts / reverse ends, or fragment starts / ends for pairs); fragment-length estimation, extension and binning are then NumPy operations (`bincount` on a difference array). The per-read Python loop reads four fields and appends an integer, which is the floor pysam allows. Chromosomes are independent and scanned in worker processes on large files (`--threads`).
- Store as per-chromosome `float32` numpy arrays. Cache to disk (npz) so re-runs skip BAM parsing entirely. This is what makes the §8 sweep affordable.

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

**Layer 2 — sub-peak detection (fine).** Scale-space decomposition of the fine-binned signal. Smooth at a bandwidth ladder (50, 150, 500, 1500, 5000 bp), find local maxima at each scale, link maxima across scales. A feature persisting across many scales is real structure; one appearing at a single scale is noise. Standard practice in astronomical source detection, and the right tool for "sharp and broad in the same track."

Three details were settled by measurement (`Benchmark/`, simulated H3K9me3-like domains with three focal peaks each):

- **Boundaries are walked on the first smoothing level, not the raw signal** (`subpeak_descent_level`). At 25 bp the raw signal has a noise wiggle every few bins; the half-maximum walk stopped at the first one and a 600 bp focal peak came back as three or four ~150 bp fragments, each too small to pass the FDR. 5 of 18 focal peaks inside domains were reported; walking the 150 bp level finds all 18. Each maximum first climbs to the top of its hill at that level, so maxima on one hill collapse into one sub-peak, and the summit is then re-read at full resolution inside the footprint.
- **Inside a domain, a sub-peak is judged against the domain's own level, not genome background** (`subpeak_vs_domain`). A sub-peak nested in a domain claims focal structure *on top of* the domain. Against genome background every bump on a 2.5× plateau is "enriched"; the 1.6× low-quality domains, which contain no focal structure at all, carried ~150 reported sub-peaks. The domain level is the median 1 kb-smoothed fold inside the region; both the fold floor and the NB test use `mu` scaled by it. This conditional test runs *without* the genome-wide autocorrelation inflation — that factor is estimated at domain scale and conditioning on the domain level is what removes it; applied anyway it is ~14× on the mixed benchmark and buries true focal peaks at −log10 p ≈ 2.
- **Candidates are local maxima, so their p-values are not uniform under the null**, and BH on them passed a third of the noise bumps. But a sample supplies its own null: nearly every candidate maximum in a domain is noise, so the median and MAD of log fold-over-domain, pooled over all domains in the sample, measure that noise directly, and a real focal peak must be an outlier against it (`subpeak_domain_z`, 4σ). Measured: every true focal peak scored z ≥ 5.1, none of ~2,400 noise candidates exceeded 3.8. Pooling is what lets a 3 kb domain with four candidates be judged at all. A noisier library widens its own null. The flat low-quality domains now carry zero sub-peaks; the 18 true ones stay.

**Merge.** Sub-peaks are assigned to their containing domain:

```
domain_00123   chr1  1,200,000  1,460,000  q=1e-31  width=260kb  n_subpeaks=4
  └ peak_00123.1  chr1  1,212,400  1,213,900  summit=1,213,050  q=1e-18
  └ peak_00123.2  ...
```

Emit `narrowPeak` (sub-peaks), `broadPeak`/`gappedPeak` (domains), plus a native nested format (BED12 or Parquet) preserving the parent/child link. Isolated sub-peaks with no enclosing domain are ordinary narrow peaks. **Nothing requires the user to declare "narrow" or "broad" up front** — that is the whole point.

### 3.3d Region-level decisions — the fold floor and the FDR scope

Two defaults decided the low-quality result, and both were wrong before being measured.

**The region fold floor was 1.5 and is now 1.2.** The local background is a *maximum* over window scales (§3.2), so it sits a little above the true level, and a 1.6× domain measures ~1.35× against it. With a 1.5 floor every one of those domains was correctly segmented by the HMM and then discarded; base-pair F1 on the low-quality benchmark was 13% at 1.5 and 96% at 1.2 with nothing else changed. The floor had been hiding exactly the marks the tool exists for. Significance, not the floor, is what keeps noise out — which is why the FDR had to change with it.

**Region FDR is taken over the genome, not over the candidates** (`fdr_scope="genome"`). The HMM scanned the whole genome before handing over a few dozen candidates; correcting over those alone is selection without paying for it. The number of tests is taken as the number of candidate-sized tiles that fit in the genome (genome size / median candidate width); the untested tiles count as null with p = 1, which leaves every rank unchanged and only scales *m*. On the null benchmark the 1.2 floor alone let 16 noise regions through; with the genome-wide count it is 2, below the 3 the old floor allowed.

**The automatic segmentation bin is floored at 100 bp** (`min_segment_bin_bp`). On a weak or overdispersed library the autocorrelation estimate (§3.3c) collapses to one bin — sparsity decorrelates a signal that is really continuous — and the HMM then decides a state per 25 bp bin, which no sparse bin can support: a 1.6× domain came back as ~50 fragments and pure noise grew a "weak" state at 1.7×. Four bins is the smallest unit that carries a decision. Measured: free on every full-size benchmark set (mixed +0.7 F1), 6× fewer false calls on a small null genome, shredded weak domains F1 0.50 → 0.78; 200 bp already costs TF-scale peaks (0.93 → 0.88).

### 3.4 The tuning objective — making "looks right in IGV" explicit

Automating parameter selection is not a search problem (grid search is trivial); it is an *objective* problem. Expert tuning by browser inspection optimizes something real but unstated: peaks land where signal visibly is, boundaries hug the enrichment, no fragmentation, no fusion. Automation requires naming that. Four objective terms, none requiring manual inspection:

1. **Held-out fit (implemented, carries most of the weight).** Split the reads at random into two halves. Call on one half; read each called piece's fold off that half; score how much better that piecewise-constant enrichment explains the *other* half's reads than background alone does (a Poisson log-likelihood ratio on data the caller never saw). This term is two-sided by construction: missing real enrichment forfeits its evidence, calling noise costs (the other half does not reproduce it), over-merging dilutes the fold over background bins which then pay the same penalty. Narrow and broad compete in the same units — evidence — so neither scale can be sacrificed for the other unnoticed.
2. **Split stability.** Calls on the two read halves agree (Jaccard). Replaces the earlier *subsampling* stability (full depth vs a 50% thinning), which was biased toward conservative settings: the thinned copy always loses the weakest calls, so the fewer weak calls a setting makes, the better it agrees with itself. On the simulated mixed benchmark that bias was decisive — the tuner chose a setting with F1 7% while scoring its stability at 0.98.
3. **Replicate reproducibility**, when true replicates are supplied.
4. **Motif central enrichment (TF mode)** and **FRiP + signal partition** — planned, not implemented.

Terms 2 and 3 are **maximized by over-merging** — one giant peak per chromosome is perfectly stable and perfectly reproducible — and term 1 alone would also accept a single giant region if the territory it swallows were nearly free. A **territory/width penalty is therefore mandatory** and the objective refuses to be constructed without one. See §12.

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

As built (`src/flexpeak/`):

```
flexpeak/
├── io/            bam.py (one-pass scan, fragment anchors, process pool), bigwig.py, load.py (format dispatch + npz cache)
├── signal/        coverage.py   — Coverage: the single in-memory representation; rebin / thin / split / subset
├── background/    nb.py         — local-lambda background with signal exclusion, NB dispersion trend, covariates
├── segment/       hmm.py (NB-HMM, dwell floor), _kernels.py (numba forward/backward/Viterbi), subpeak.py (scale space)
├── stats/         nbtest.py     — NB tails, aggregate dispersion, autocorrelation, BH in log space
├── tune/          objective.py (held-out fit, stability, territory), sweep.py (grid, read halves, plateau)
├── qc/            metrics.py
├── plots/         style.py, stats.py
├── caller.py      CallParams (every default documented with its measurement), call_peaks
├── peaks.py       NestedPeakSet / Region / SubPeak and the four output formats
├── simulate.py    synthetic data with exact truth — the test suite's and the benchmark's substrate
└── cli.py         callpeak | tune | qc | stats
```

Not built: `annotate/` (§6), the HTML report (§5), `data/` (bundled blacklists and chrom sizes). Interval operations are a few hundred lines in `peaks.py` rather than a dependency; the planned pyranges/bioframe dependency was not needed for what the caller does.

**CLI** — deliberately MACS2-shaped so people can switch in one line:

```bash
flexpeak callpeak -t chip.bam -c input.bam -n H3K9me3_sample          # self-tunes, nested output
flexpeak callpeak -t sig.bw --preset h3k27me3 --no-tune -n K27        # bigWig input, preset, bit-identical reruns
flexpeak callpeak -t chip.bam --threads 8 --blacklist mm10.bed.gz     # parallel BAM scan, exclusion list
flexpeak tune     -t chip.bam --chroms chr1,chr19                     # inspect the objective curve
flexpeak qc       -t chip.bam -p peaks.nested.tsv
flexpeak stats    -p peaks.nested.tsv --outdir figures                # width/count/enrichment figures, PDF
```

**Dependencies:** numpy and scipy only, at the core. Optional extras: pysam (BAM/CRAM), pyBigWig (bigWig), numba (HMM kernels; a NumPy fallback is tested to agree), matplotlib (figures). No scikit-learn, no LightGBM, no torch. Install is `pip install flexpeak`, full stop — no model download, no checksums, no version pinning of weights.

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

**Measured** (`Benchmark/`, 4.5 Mb synthetic genome, BAM parse charged to both, single-threaded): FlexPeak 2.9 s against MACS2 7.5 s and MACS3 8.1 s — **2.6× faster**, down from 1.8× slower before the work below; the self-tuning sweep (27 grid points, two read halves) adds ~5 s. Where the time went, in order of what was fixed:

1. **One pass over the BAM, no per-read Python.** The loader scanned every read in a Python loop and then made a *second* full pass for the fragment-length estimate, indexing a NumPy array once per read. Both passes are now one scan into integer anchor arrays, and everything downstream is `bincount`. Load time for a treatment/control pair: 11.7 s → 2.7 s, coverage bit-identical.
2. **The HMM emission's gamma-function coefficient** depends on counts and dispersion but not on the state multiplier; it was recomputed per state per EM iteration and was half the cost of a whole call. Computed once per chromosome now.
3. **Sub-peak candidates are tested in one vectorised NB call** from prefix sums; a large domain can hold thousands and testing them one by one was ~90% of a call on mixed data.
4. **The tuner reuses what the grid does not change**: the background is fitted once per read half, and the HMM once per half per segmentation setting — the posterior cutoff and fold floor only threshold its output. 53 s → 7.6 s.
5. **Per-chromosome multiprocessing** exists in the loader (`load_bam(n_jobs=)`, `--threads`) and is automatic above 10 M reads; below that the worker start-up costs more than the scan. Under the `spawn` start method (macOS, Windows) it is opt-in for library callers, because a script without a `__main__` guard would re-run itself in every worker.
6. **Bin, don't iterate per-bp;** **numba** for forward–backward and Viterbi, with a NumPy fallback tested to agree; **cache the binned coverage** so re-runs skip BAM I/O; **stratified chromosome subset** for tuning (one large, one mid, one small); CRAM and bigWig input.

---

## 8. Self-tuning at runtime — no training corpus

The tuner learns from the sample in front of it. This removes the largest cost, the largest risk, and the largest reviewer objection from the original plan simultaneously.

### 8.1 The insight

A sample can validate itself. Split its reads at random into two halves: each half is an independent draw of the same underlying signal, so calls made on one half can be tested against reads the caller never saw.

1. Load the treatment BAM, build binned coverage on the stratified chromosome subset (§7), cache it.
2. Split the binned counts binomially into two halves (`Coverage.split`); more seeds give more splits.
3. Sweep the parameter grid (27 points: posterior cutoff × fold floor × segmentation scale). For each setting, call on each half.
4. Score with the §3.4 objective: held-out fit of each half's calls against the other half's reads, agreement between the halves, territory penalty.
5. Take the plateau. Report the choice and the curve.

The original design thinned the sample to 50% and 25% and compared those calls with full-depth calls ("every sample is deeper than a subsample of itself"). That is true, and it was the wrong comparison: the thinned copy always loses the weakest calls, so the fewer weak calls a setting makes, the better it agrees with itself, and the tuner was rewarded for calling almost nothing (§3.4). Two halves are symmetric, and neither is the reference.

Coverage is parsed once; the background is fitted once per half; the HMM is fitted once per half per segmentation setting and re-thresholded at every grid point. The sweep costs ~5 s on the benchmark genome.

**What the tuner can and cannot do today.** On every benchmark dataset the tuned setting reproduces the defaults within noise, which is the right answer there and the opposite of the previous behaviour. It does not yet improve on the defaults: on a shallower low-quality simulation where a 500 bp segmentation would lift F1 from 0.78 to 0.91, the half-depth objective cannot distinguish the two and keeps the default. An absolute segmentation-bin axis for the grid was measured and rejected — it rescues that case only in some configurations and costs TF-like data 5–15 F1 points, because the held-out term barely penalises boundary bloat. Sharpening the objective to boundary precision is the open problem.

### 8.2 Why this is better, not merely cheaper

- **No training-set bias** to defend — no cell-type imbalance, no mark imbalance, no ChIP→CUT&RUN domain shift, no "does it generalize to my organism."
- **Works out of distribution by construction:** novel marks, non-model organisms, new assay chemistries, unusual protocols.
- **No model versioning, download, or optional extras.**
- **Fully auditable:** the decision is a scored curve over an explicit grid, not a learned function.
- **"No training data required" is a selling point** against LanceOtron and similar, not a weakness.

### 8.3 Failure modes to guard

- **Over-merge drift** (§3.4) — the objective's built-in bias. Mandatory territory penalty; validate specifically on broad marks, where the pull is strongest.
- **Under-call collapse** — the symmetric failure, which the original objective did not guard and which happened (F1 7% on mixed data). The held-out fit term is the guard; a strict setting forfeits the evidence of everything it drops.
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
| Plateau selection beats argmax | Select argmax instead | Higher variance across split seeds |
| Held-out fit is what makes the objective two-sided | `TuningObjective(heldout_weight=0)` | Tuner collapses to a conservative corner on mixed data (measured: F1 99% → 7%) |
| Domain-level reference for sub-peaks | `subpeak_vs_domain=False` | Flat noisy domains fill with noise sub-peaks (measured: 0 → ~150) |
| Smoothed boundary descent | `subpeak_descent_level=0` | Focal peaks inside domains fragment below the FDR (measured: 18/18 → 5/18) |
| Genome-scoped region FDR | `fdr_scope="candidates"` | Null false calls rise (measured: 2 → 16 at the 1.2 fold floor) |
| Segmentation-bin floor | `min_segment_bin_bp=25` | Weak domains shred on shallow data (measured: F1 0.78 → 0.50) |

The last five are implemented as `CallParams` switches, run in `tests/test_tune_and_calls.py`, and measured on the simulated benchmark; the numbers quoted are from `Benchmark/README.md`.

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
| **P3** | Self-tuning: read-split validation, two-sided objective, sweep, plateau selection, presets. **Gate: recovers §3.4 expert parameters unseen** | 2–3 weeks |
| **P4** | QC + HTML report + annotation. **Gate: the degradation-curve figure** | 3–4 weeks |
| **P5** | Held-out evaluation, docs, bioconda/PyPI, tutorials, preprint | 4–5 weeks |

P0–P2 is already a publishable, useful tool. P3 adds adaptivity with no training data and no ML dependency.

**Status.** P0–P3 are implemented (P0 without CI; the evaluation harness exists for simulated data only). P3's gate — recovering the expert parameters in §3.4 unseen — has not been run; the tuner's current measured state is §8.1. P4 is partial (QC metrics and statistics figures exist; no HTML report, no annotation). P5 has not started.

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

## 14. Next steps

Done from the original list: name, license (MIT), `pyproject.toml` + `src/` layout + pytest, the `NestedPeakSet` contract, the simulated harness with MACS2/MACS3 wrappers, P1–P3.

1. **Experimental-data evaluation (§9–10).** Everything quantitative so far is on simulation drawn from the model family FlexPeak assumes. Assemble the §9.3 datasets, sequester the held-out set, and run R4 (input-vs-input) first — it is the cheapest test and the one the NB claim rests on.
2. **The §3.4 expert-parameter gate** for the tuner on the in-house CUT&RUN set (`Code/legacy/`).
3. **Sharpen the tuning objective to boundary precision** (§8.1): the held-out term does not yet pay enough for flank bloat to let the grid carry an absolute scale axis.
4. **Real comparators**: epic2/SICER2 and SEACR in an environment where they build; HMMRATAC as the closest precedent.
5. CI on the test suite; GitHub Actions; bioconda recipe.
6. P4: HTML QC report; annotation (§6).
