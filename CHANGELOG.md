# Changelog

## 1.1.0 — improve-algorithm-speed

- Parallelize chromosome-level background fitting and peak calling, with
  segmentation caches reused across workers and tuning runs.
- Improve sparse-library background estimation and ATAC peak calling; add
  configurable dispersion trends and HMM fitting on a fraction of the data.
- Reduce coverage-cache storage, reuse whole-genome caches for chromosome
  subsets, and remove run caches by default (`--keep-cache` retains them).
- Expand ATAC quality control, including TSS enrichment, and provide a
  single-page QC figure by default with `--figures`.
- Refresh the README, algorithm documentation, tutorial, and benchmark
  material; move the GEO example into `examples/` and remove cluster-specific
  scripts and generated experimental-data figures from the package repository.
- Add regression coverage for sparse libraries, parallel calling, cache
  handling, HMM fitting, and QC output.
