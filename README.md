# FlexPeak

Flexible, adaptive peak calling for ChIP-seq / CUT&RUN / CUT&Tag / ATAC-seq.

No pretrained model, no training corpus, no ML dependency — adaptivity comes from
per-sample self-tuning at runtime. Narrow peaks and broad domains come out of the
same run, nested, without being asked to declare which you expect.

```bash
pip install -e ".[bigwig,figures,fast]"

flexpeak callpeak -t chip.bam -c input.bam -n H3K9me3_sample
flexpeak callpeak -t signal.bw --preset h3k27me3 --no-tune --figures -n K27
```

## Documentation

- **[docs/TUTORIAL.md](docs/TUTORIAL.md)** — start here. A reproducible
  walkthrough on real mm10 CUT&RUN data: first call in a minute, reading the
  outputs and QC, inspecting a locus, and what to do when a result looks wrong.
- [docs/DESIGN.md](docs/DESIGN.md) — the algorithm and the reasoning behind it.
- [testing/README.md](testing/README.md) — the real-data integration test.
- [examples/figures.py](examples/figures.py) — correctness figures on simulated
  data with known truth.

## Output

Every run writes four projections of one nested call set — `narrowPeak`,
`broadPeak`, BED12 (nesting rendered for IGV), and a native TSV that round-trips
— plus a QC report and, with `--figures`, eight statistics figures as vector PDFs.
