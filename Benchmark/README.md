# FlexPeak method comparison

A reproducible comparison of FlexPeak against other peak-calling approaches on
simulated data with exact ground truth.

```bash
python Benchmark/run_benchmark.py     # ~7 min: run every method on every dataset
python Benchmark/make_figures.py      # render Benchmark/figures/
```

> **This is not the benchmark specified in [docs/DESIGN.md §9–10](../docs/DESIGN.md).**
> That one requires experimental data, held-out datasets and the published
> software for every comparator. This is a controlled comparison on simulation,
> and half the comparators are re-implementations rather than the real tools
> (§2). It is a regression harness and a sanity check, not evidence for a paper.

---

## 1. Headline results

Base-pair F1 against known truth, all methods at their own nominal FDR 5%:

| Method | narrow | broad | mixed | low quality |
|---|---:|---:|---:|---:|
| **FlexPeak (defaults)** | 91% | **99%** | **99%** | 58% |
| FlexPeak (self-tuned) | 91% | **99%** | 7% | 8% |
| FlexPeak (preset h3k27me3) | 56% | 95% | 97% | 64% |
| MACS2 / MACS3 narrow | 90% | 52% | 55% | 15% |
| MACS2 / MACS3 broad | 80% | 94% | 94% | 44% |
| SICER/epic2-style, defaults | 11% | 94% | 95% | 54% |
| SICER/epic2-style, tuned | 17% | 97% | 97% | 63% |
| DESeq2/csaw-style, 200 bp | 61% | 0% | 8% | 0% |
| DESeq2/csaw-style, 5 kb | 21% | 98% | 92% | 56% |
| FlexPeak, Poisson background (ablation) | 69% | 96% | 91% | 52% |

Four results matter, and two of them are unfavourable to FlexPeak:

1. **FlexPeak handles all three scales with one parameter set.** Every other
   method has a configuration that is good on narrow *or* broad and bad on the
   other. This is the design claim, and on this data it holds.
2. **FlexPeak's boundaries are an order of magnitude tighter** on broad
   domains: median |Δstart| 312 bp, against 18,944 bp for MACS broad, 1,112 bp
   for the tuned SICER-style baseline and 1,075 bp for 5 kb NB windows.
3. **The self-tuner is actively harmful on two of four datasets** — F1 collapses
   from 99% to 7% on mixed and from 58% to 8% on low-quality data (§5.1). Plain
   defaults are far better. This is a defect, not a caveat.
4. **FlexPeak is not the best method on low-quality data.** The tuned
   SICER-style baseline (63%) and 1 kb windows (66%) beat FlexPeak's defaults
   (58%), which is the dataset FlexPeak's stated motivation targets most
   directly.

![F1 heatmap](figures/B1_f1_heatmap.png)

## 2. What was actually run, and what was not

| Comparator | Status |
|---|---|
| **MACS2 2.2.9.1** | Real tool, executed as a subprocess. Version and command line recorded. |
| **MACS3 3.0.0b3** | Real tool, executed as a subprocess. Identical results to MACS2 on every dataset here. |
| **SICER2 / epic2** | **Not run.** `pip install epic2 SICER2` fails on this machine: `epic2`, `ncls` and `sorted_nearest` cannot build C extensions. Replaced by a re-implementation of the island algorithm. |
| **DESeq2** | **Not run.** `BiocManager::install("DESeq2")` fails on R 4.1.3 here. Replaced by a re-implementation of windowed NB testing. |
| **SEACR** | **Not run.** Fetching the upstream script was blocked in this environment. No substitute implemented. |
| **HMMRATAC, LanceOtron, Genrich, JAMM** | Not attempted. |

**The two re-implemented baselines are labelled `(re-impl.)` everywhere and are
not evidence about SICER2, epic2 or DESeq2.** They implement the published
*algorithm*; they are not version-matched, not author-tuned, and any difference
may be an artefact of this implementation. In particular the SICER-style
baseline omits the null-derived island-score threshold that real SICER applies
before FDR control, which is very likely why its null-dataset specificity here
(§5.3) is worse than the published tools would be.

Both baselines deliberately reuse FlexPeak's NB arithmetic
(`flexpeak.stats.nbtest`) where NB testing is involved. That is intentional:
holding the arithmetic fixed isolates the *structural* difference — HMM
segmentation, local-lambda background, scale-space sub-peaks — instead of
confounding it with a second implementation of the same maths.

## 3. Protocol

### 3.1 Data

Five simulated datasets, 4.5 Mb genome (chr1 3 Mb + chr2 1.5 Mb), 25 bp bins,
negative-binomially overdispersed background with drift shared between
treatment and control. Ground truth is exact by construction.

| Dataset | Content | Truth regions |
|---|---|---:|
| `narrow` | focal peaks, 8× enrichment (TF / H3K4me3-like) | 40 |
| `broad` | broad domains, 2.5× (H3K27me3-like) | 12 |
| `mixed` | narrow + broad + domains containing sub-peaks (H3K9me3-like) | 74 |
| `lowqual` | 3× narrow / 1.6× broad, dispersion 0.40 | 86 |
| `null` | no enrichment anywhere | 0 |

Every method reads the **same BAM files** and the same control. FlexPeak's
coverage cache is disabled and the BAM parse time is added to its wall clock,
because MACS re-parses the BAM on every invocation; charging FlexPeak nothing
for I/O would flatter it (§5.5).

### 3.2 Fairness measures

Following [DESIGN.md §10.2](../docs/DESIGN.md):

- All methods run at nominal FDR 5%. Matched call count is not enforced;
  territory is reported alongside every recall number instead.
- MACS runs in **both** narrow and broad modes on every dataset, so it is never
  scored only in the mode that suits the data.
- Each re-implemented baseline runs in **three** configurations — its defaults
  and two mark-appropriate ones. Comparing a self-tuned method against
  fixed-parameter comparators would be a strawman.
- `--nomodel --extsize 50` is given to MACS because the simulated reads carry no
  strand asymmetry for its cross-correlation model to estimate from. 50 bp is
  the true simulated fragment length — the same quantity FlexPeak derives from
  the same file. Neither method is advantaged.
- Every method is scored by the same code (`metrics.py`) from call intervals
  alone. No metric reads any method's score column.

### 3.3 Metrics

Base-pair precision / recall / F1 and Jaccard on territory; region recall (a
truth region counts as recovered at ≥40% base-pair coverage); **fragmentation
index** (truth regions split across >1 call); **fusion index** (calls spanning
>1 truth region); median boundary error; genome fraction called; wall clock.

Fragmentation and fusion are the two boundary failure modes and are only
meaningful together — a method can score well on one by failing at the other.

## 4. Figures

![precision vs recall](figures/B2_precision_recall.png)

![fusion vs fragmentation](figures/B3_fusion_vs_fragmentation.png)

![recall vs territory](figures/B5_recall_vs_territory.png)

![null specificity](figures/B4_null_specificity.png)

![runtime](figures/B6_runtime.png)

## 5. Findings

### 5.1 The self-tuner degrades results on hard data — a real defect

| Dataset | Defaults F1 | Self-tuned F1 | Self-tuned territory |
|---|---:|---:|---:|
| narrow | 91% | 91% | 1.2% |
| broad | 99% | 99% | 16.0% |
| **mixed** | **99%** | **7%** | 0.7% (vs 19.3%) |
| **low quality** | **58%** | **8%** | 0.5% (vs 4.2%) |

On mixed and low-quality data the tuner selects a setting whose base-pair recall
is 4%, with precision still high. It is not over-merging — it is collapsing to
an extremely conservative corner, keeping only the strongest focal peaks and
discarding essentially all domain territory.

The mechanism is visible in the objective (DESIGN.md §3.4). It is

```
objective = mean(stability, reproducibility) - territory_penalty
```

Stability is the Jaccard between full-depth and subsampled calls. A very strict
setting produces few, strong, perfectly reproducible calls, so stability
approaches 1. The territory penalty only engages *above* a genome-fraction cap,
so it does nothing to punish under-calling. **The objective is one-sided:** the
design correctly identified that stability is maximised by over-merging and
added a penalty for it, but nothing counterbalances the symmetric degenerate
solution of calling almost nothing.

This is a design problem in the objective, not a tuning-grid problem, and it is
why `--no-tune` currently outperforms the tuner on exactly the data the tuner
exists to help with.

### 5.2 MACS in narrow mode fragments broad domains, as documented

On `broad`, MACS2/3 narrow reports 1,819 calls for 12 true domains: fragmentation
index 100%, region recall 17%, median |Δstart| 42,796 bp. In broad mode the same
tool gets F1 94% but still fragments every domain (fragmentation 100%,
|Δstart| 18,944 bp) — it recovers the territory as a scatter of sub-peaks rather
than as domains. FlexPeak's fragmentation index is 0% with |Δstart| 312 bp.

This is the clearest result in the comparison and it is what the nested output
is for.

### 5.3 Specificity on the null dataset

| Method | calls | genome called |
|---|---:|---:|
| MACS2/3 narrow | 18 | 0.03% |
| FlexPeak (defaults) | 10 | 0.07% |
| MACS2/3 broad | 121 | 0.40% |
| FlexPeak (self-tuned) | 94 | 0.54% |
| **FlexPeak, Poisson ablation** | 827 | **2.58%** |
| FlexPeak (preset h3k27me3) | 288 | 7.34% |
| SICER-style (re-impl.) | 286–745 | 24–32% |
| DESeq2/csaw-style (re-impl.) | 0 | 0.00% |

The correct answer is zero. Two things to read carefully here:

- **The simulator draws background from the negative binomial family FlexPeak
  assumes**, so FlexPeak is favoured by construction. The comparison that does
  *not* depend on that assumption is FlexPeak against its own Poisson ablation:
  identical code, one distribution swapped, 0.07% → 2.58%. That is a controlled
  37-fold difference and it is the strongest support here for the NB claim.
- **The SICER-style figure should not be attributed to SICER2 or epic2** (§2).
  A Poisson island test on overdispersed background is genuinely
  anti-conservative, but the published tools apply a null-derived score
  threshold this re-implementation lacks.
- `--preset h3k27me3` calls 7.34% of a null genome, worse than the Poisson
  ablation. This corroborates the main README's statement that the presets are
  uncalibrated placeholders and should not be used.

### 5.4 Where FlexPeak loses

- **Low-quality data**: F1 58% for FlexPeak defaults against 66% for the 1 kb
  SICER-style baseline and 63% for the tuned one. This is the scenario
  FlexPeak's motivation targets most directly, and it does not win it.
- **Narrow precision**: 86% for FlexPeak against 95% for MACS narrow at
  comparable territory. MACS is the more precise focal caller here.
- **Broad domains, 5 kb NB windows**: F1 98% against FlexPeak's 99% — a
  windowed NB test with a well-chosen window size is nearly as good on pure
  broad data, and far simpler. FlexPeak's advantage over it appears on `mixed`
  (99% vs 92%) and in boundary error, not on `broad` alone.

### 5.5 Runtime: FlexPeak is slower than MACS here

End to end, with BAM parsing charged to both:

| Method | median wall clock |
|---|---:|
| FlexPeak (defaults) | 12.6 s |
| MACS2 narrow / broad | 6.8 / 6.7 s |
| MACS3 narrow / broad | 7.3 / 7.2 s |
| FlexPeak (self-tuned) | 53.2 s |

FlexPeak is **~1.8× slower than MACS2** on this genome, against the ≤1.5×
target in [DESIGN.md §7](../docs/DESIGN.md). The self-tuner costs a further
30–55 s. Most of FlexPeak's time is the pysam read loop, not the HMM: the
Poisson ablation, which skips nothing but the NB dispersion fit and runs on
coverage already in memory, takes 2.0 s.

Two reasons not to read much into this: a 4.5 Mb genome is far below the scale
where the per-chromosome parallelism in DESIGN.md §7 would matter (and it is
not implemented), and the re-implemented baselines are excluded from this
comparison entirely because they never touch a BAM.

## 6. Threats to validity

1. **Simulated data only.** The background is drawn from FlexPeak's own model
   family. This inflates FlexPeak's null-dataset performance and may inflate its
   power. No conclusion here transfers to experimental data.
2. **Two comparators are re-implementations** (§2), and one of them
   (SICER-style) is measurably harsher on itself than the real tool would be.
3. **SEACR was not run at all**, so the no-control CUT&RUN case is untested.
4. **One replicate per condition.** No seed variation, so none of these numbers
   carry an uncertainty estimate. Differences under a few percent are not
   interpretable.
5. **A 4.5 Mb genome** is three orders of magnitude smaller than a real one. The
   runtime figure is a smoke test; it says nothing about scaling.
6. **The evaluation code is part of the FlexPeak repository.** It was written
   after seeing FlexPeak's behaviour, which is a real source of metric-choice
   bias even though every method is scored identically.

## 7. Files

```
Benchmark/
├── run_benchmark.py            simulate -> run every method -> metrics.tsv
├── methods.py                  method wrappers; tools vs re-implementations
├── metrics.py                  all evaluation metrics, one scorer for all
├── make_figures.py             renders figures/ from results/metrics.tsv
├── data/                       simulated BAMs (regenerated if deleted)
├── results/
│   ├── metrics.tsv             one row per (dataset, method), all metrics
│   ├── commands.txt            exact command line for every run
│   ├── versions.txt            tool versions and environment
│   ├── calls/*.bed             every method's calls, exactly as scored
│   └── macs_raw/               unmodified MACS output
└── figures/B1..B6*.png
```

To add a comparator, add a wrapper to `methods.py` returning
`{"intervals", "runtime_s", "command", "note"}` and register it in
`available_methods()` in `run_benchmark.py`. Nothing else needs to change.
