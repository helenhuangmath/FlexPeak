#!/bin/bash
source /vast/projects/wherry/foundation-models-immuno/hhua/github/FlexPeak/FlexPeak/cluster/_env.sh
D=$REPO/Benchmark/data_diff; O=$REPO/tmp/tooldbg; rm -rf $O; mkdir -p $O
echo "### THOR"
cat > $O/thor.config <<CFG
#rep1
$D/diff_mixed_A1.bam
$D/diff_mixed_A2.bam
#rep2
$D/diff_mixed_B1.bam
$D/diff_mixed_B2.bam
#chrom_sizes
$D/diff_mixed.chrom.sizes
CFG
cat $D/diff_mixed.chrom.sizes
rgt-THOR $O/thor.config --output-dir $O/thor_out --name t --pvalue 0.05 --no-correction 2>&1 | tail -30
ls $O/thor_out 2>/dev/null
echo "### sicer_df"
cd $O
bedtools bamtobed -i $D/diff_mixed_A1.bam > A.bed; bedtools bamtobed -i $D/diff_mixed_B1.bam > B.bed; head -2 A.bed
sicer_df -t A.bed B.bed -s mm10 -w 200 -g 600 -f 50 -egf 0.0022 -rt 1 -fdr_df 0.05 -e 1000 -o $O -cpu 1 2>&1 | tail -25
ls $O | head -20
