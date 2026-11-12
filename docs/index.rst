FlexPeak
========

**Flexible, adaptive peak calling for ChIP-seq, CUT&RUN, CUT&Tag and ATAC-seq.**

Introduction
------------

FlexPeak identifies enriched regions in chromatin profiling data. Narrow peaks
and broad domains are called in a single run and reported as one *nested* call
set, in which focal sub-peaks are placed inside the domains that contain them.
The type of mark (narrow, broad or mixed) does not have to be declared in
advance.

The method combines a negative binomial background model, a three-state hidden
Markov model for segmentation, and region-level significance testing with
false discovery rate control. Parameters are adapted to each sample by a
tuning sweep that validates candidate settings on held-out halves of the
sample's own reads; no pretrained model or training corpus is used.

Main features
-------------

* **One call set for every mark.** Transcription factors, H3K4me3, H3K27ac,
  H3K27me3, H3K9me3, H3K36me3 and ATAC-seq are handled by the same model;
  domains carry their sub-peaks.
* **Self-tuning.** A held-out read split chooses the settings for each
  sample; presets and ``--no-tune`` give bit-identical, untuned runs.
* **With or without a control.** The background is estimated from an input
  control or, for CUT&Tag, CUT&RUN and ATAC-seq, from the treatment with
  apparent signal excluded.
* **Low-depth libraries.** The background adapts to sequencing depth, so
  sparse libraries are not reduced to a few very wide calls.
* **Differential calling** between two conditions, with or without
  replicates (``flexpeak diff``).
* **Quality control.** Library metrics, a quality grade with its reasons,
  FRiP, and for ATAC-seq the TSS enrichment and fragment-size ladder, in one
  report and one optional page of figures.
* **Fast and light.** Chromosomes are processed in parallel; the core
  depends only on NumPy and SciPy, and the temporary coverage cache is
  removed when the run finishes.

Supported input
---------------

Coordinate-sorted, indexed **BAM** or **CRAM** files (single- or paired-end)
and **bigWig** signal tracks. Output is written as ENCODE narrowPeak and
broadPeak, BED12 and a native nested table. See :doc:`input_output`.

Getting help
------------

Questions and bug reports are welcome on the
`GitHub issue tracker <https://github.com/helenhuangmath/FlexPeak/issues>`_.
``flexpeak <command> --help`` lists every option of a command.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart
   TUTORIAL

.. toctree::
   :maxdepth: 2
   :caption: User guide

   commands
   input_output
   parameters
   atac
   differential
   cli

.. toctree::
   :maxdepth: 2
   :caption: Methods and evaluation

   DESIGN
   benchmark

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   citation
