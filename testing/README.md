# Real-Data Test

This directory keeps one integration test against public real data. The data is
not stored in the repository; the test downloads it on demand into ignored
working directories.

## Source

The test sample is from NCBI GEO accession **GSE285245**:

```text
https://ftp.ncbi.nlm.nih.gov/geo/series/GSE285nnn/GSE285245/suppl/GSE285245_pool_H3K9me3_Cl13.bw
```

It is an mm10 H3K9me3 bigWig. FlexPeak treats bigWig signal as normalized signal
rescaled to pseudo-counts, so p/q-values are approximate; use the run for
pipeline and shape sanity checks, not as a calibrated statistical benchmark.

## Run

```bash
python testing/run_geo_h3k9me3.py --quick
```

By default `--quick` runs chr19 only and writes temporary outputs under
`testing/results/geo_h3k9me3/`, including the statistics figures as PDFs in
`figures/`. Drop `--quick` to call chr1-19 plus chrX.

Add `--keep-figures` to also write the tutorial figures (summary, width,
enrichment, width vs enrichment) as PNGs into `docs/figures/tutorial/`. That
directory is tracked by git, and [docs/TUTORIAL.md](../docs/TUTORIAL.md) embeds
those images. Use `--no-figures` to skip figures, for example on a machine
without matplotlib.

The test asserts that the run completes, output projections are written, the
native TSV round-trips, regions are sorted/non-overlapping, and QC values are in
a sane range.
