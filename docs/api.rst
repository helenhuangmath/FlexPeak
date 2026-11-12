Python API
==========

The main functions and classes are importable from the top-level package.

.. code-block:: python

   from flexpeak import load_signal, call_peaks, tune, CallParams

Loading signal
--------------

.. autofunction:: flexpeak.load_signal

.. autoclass:: flexpeak.Coverage
   :members: rebin, subset, thin, split

Calling peaks
-------------

.. autofunction:: flexpeak.call_peaks

.. autoclass:: flexpeak.CallParams

.. autofunction:: flexpeak.preset_params

.. autoclass:: flexpeak.BackgroundModel
   :members: fit

Self-tuning and scanning
------------------------

.. autofunction:: flexpeak.tune

.. autoclass:: flexpeak.TuneResult
   :members:

.. autoclass:: flexpeak.TuningObjective

.. autofunction:: flexpeak.scan_file

.. autofunction:: flexpeak.scan_coverage

.. autoclass:: flexpeak.ScanProfile

Call sets
---------

.. autoclass:: flexpeak.NestedPeakSet
   :members:

.. autoclass:: flexpeak.Region

.. autoclass:: flexpeak.SubPeak

.. autofunction:: flexpeak.peaks.read_nested_tsv

Simulation
----------

.. autofunction:: flexpeak.simulate.simulate
