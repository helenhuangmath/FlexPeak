# Differential calling

```bash
flexpeak diff -a KO_rep1.bam KO_rep2.bam -b WT_rep1.bam WT_rep2.bam -n KO_vs_WT
flexpeak diff -a Arm.bw -b Cl13.bw --chroms chr1,chr2 -n Arm_vs_Cl13   # single samples
```

## Method

Samples are normalised by matched background level by default; `--norm tmm`
and `--norm total` are the alternatives. Replicates are averaged within each
condition. In each bin, the proportion of signal from condition A is modelled
by a three-state HMM (unchanged, higher in A, higher in B) with beta-binomial
emissions. The overdispersion is estimated from replicates where they exist,
and within the EM otherwise.

Candidate regions are tested on their aggregate counts with a region-level
overdispersion, followed by Benjamini–Hochberg correction. A candidate must
also be enriched in at least one condition, over the control or over the
local background of the pooled signal. `--within-peaks` restricts calls to
peaks found in either condition.

## Output

| File | Contents |
|---|---|
| `<name>_diffpeaks.narrowPeak` | differential regions; strand `+` for higher in A and `-` for higher in B; the signal value is the log2 fold change |
| `<name>_diffpeaks.tsv` | counts per replicate, posterior, p- and q-values |
| `<name>_gain.bed`, `<name>_loss.bed` | regions higher in A and higher in B |
