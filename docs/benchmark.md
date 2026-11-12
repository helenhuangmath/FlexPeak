# Benchmark

The repository contains a reproducible comparison on simulated data with
known ground truth. MACS2, MACS3, SICER2 and SEACR are run as external
programs on the same alignments and the same control, each at its nominal 5 %
false discovery rate.

```bash
python Benchmark/run_benchmark.py     # writes Benchmark/results/metrics.tsv
python Benchmark/make_figures.py      # writes Benchmark/figures/
```

## Results

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

```{figure} ../Benchmark/figures/B1_f1_heatmap.png
:width: 100%
:alt: Base-pair F1 by method and dataset

Base-pair F1 for every method and setting on the five simulated datasets.
```

## Limitations

These are simulation results. The simulated background is drawn from the
model family that FlexPeak assumes, which favours it on the null dataset.
MACS2 in narrow mode is slightly more precise on focal peaks, and SICER2
recovers slightly more broad territory. On these datasets the self-tuning
sweep reproduces the defaults rather than improving on them.

The protocol, the differential comparison and a full discussion are in
[Benchmark/README.md](https://github.com/helenhuangmath/FlexPeak/blob/main/Benchmark/README.md).
