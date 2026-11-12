# ATAC-seq

ATAC-seq libraries are usually run without a control, are paired-end, and are
often sequenced more shallowly than ChIP-seq. The `atac` preset and the ATAC
quality-control metrics are designed for these libraries.

```bash
flexpeak callpeak -t atac.bam --preset atac --no-tune --threads 8 \
    --blacklist hg38-blacklist.v2.bed.gz --tss gencode.gtf --figures -n sample
```

## What the preset changes

- **Resolution.** 10 bp bins and 1–10 kb background windows, which suit
  accessible elements of a few hundred base pairs.
- **Width cap.** Regions wider than 50 kb (`--max-width`) are reported as
  their sub-peaks. Accessible elements, and clusters of them, are far narrower
  than this, so the cap only applies to stretches such as amplified regions of
  an aneuploid genome.
- **Sampled HMM fit.** The expectation–maximisation iterations of the HMM run
  on a 25 % sample of each chromosome, and the posterior is then computed on
  the whole chromosome (`--hmm-fit-fraction`). On test libraries this
  changed the regions by less than 1.5 % (Jaccard ≥ 0.988) and shortened
  segmentation by about half.
- **Depth-aware background.** As for all count data, background windows are
  widened in sparse libraries, so a library with a few million fragments is
  still fitted on real background rather than on empty bins.

## Quality control

With `--preset atac` the report uses metrics suited to ATAC-seq:

| Metric | Source | Notes |
|---|---|---|
| TSS enrichment | `--tss` (GTF, GFF or BED) | signal at transcription start sites relative to their flanks |
| Nucleosome-free fraction | fragment sizes | fragments shorter than one nucleosome |
| Mono-nucleosome fraction | fragment sizes | fragments spanning one nucleosome |
| FRiP | call set | fraction of fragments in called regions |

The normalised strand cross-correlation coefficient (NSC) is reported for
single-end data only. Low NSC is not reported as a failed library for an
assay without immunoprecipitation.
