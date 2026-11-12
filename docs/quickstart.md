# Quick start

## ChIP-seq with an input control

Peak calling on a treatment BAM with an input control, using self-tuning:

```bash
flexpeak callpeak -t chip.bam -c input.bam -n sample --outdir out/
```

## CUT&RUN or CUT&Tag without a control

Without a control, the background is estimated from the treatment with
apparent signal excluded:

```bash
flexpeak callpeak -t cutntag.bam -n sample
```

## From a bigWig, with a preset

With a mark preset and no tuning sweep, repeated runs are bit-identical:

```bash
flexpeak callpeak -t signal.bw --preset h3k27me3 --no-tune -n K27
```

## Several workers, an exclusion list and the QC page

```bash
flexpeak callpeak -t chip.bam -c input.bam --threads 8 \
    --blacklist mm10-blacklist.v2.bed.gz --figures -n sample
```

## ATAC-seq

Paired-end ATAC-seq without a control, with ATAC-specific quality control
(TSS enrichment needs a gene annotation; see {doc}`atac`):

```bash
flexpeak callpeak -t atac.bam --preset atac --no-tune --threads 8 \
    --blacklist hg38-blacklist.v2.bed.gz --tss gencode.gtf --figures -n sample
```

## What a run writes

Each run writes the call set in four formats plus a QC report (see
{doc}`input_output`). The tuning report, which lists the settings evaluated
and the one selected, is printed to standard error and recorded in the output
headers. The temporary coverage cache is deleted when the run finishes.

## From Python

```python
from flexpeak import load_signal, call_peaks, tune, CallParams

treat = load_signal("chip.bam", bin_size=25)
ctrl  = load_signal("input.bam", bin_size=25)

result = tune(treat, ctrl, base=CallParams())   # TuneResult: params, curve, report()
print(result.report())

peaks = call_peaks(treat, ctrl, result.params, n_jobs=8)   # NestedPeakSet
peaks.write_all("out/sample")                              # the four call-set files

for region in peaks:
    print(region.chrom, region.start, region.end, region.kind,
          region.fold_enrichment, region.neg_log10_q, len(region.subpeaks))
```

See {doc}`api` for the full Python interface.
