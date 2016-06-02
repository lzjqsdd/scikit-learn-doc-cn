

.. _example_decomposition_plot_ica_blind_source_separation.py:


=====================================
Blind source separation using FastICA
=====================================

An example of estimating sources from noisy data.

:ref:`ICA` is used to estimate sources given noisy measurements.
Imagine 3 instruments playing simultaneously and 3 microphones
recording the mixed signals. ICA is used to recover the sources
ie. what is played by each instrument. Importantly, PCA fails
at recovering our `instruments` since the related signals reflect
non-Gaussian processes.




.. image:: images/plot_ica_blind_source_separation_001.png
    :align: center




**Python source code:** :download:`plot_ica_blind_source_separation.py <plot_ica_blind_source_separation.py>`

.. literalinclude:: plot_ica_blind_source_separation.py
    :lines: 16-

**Total running time of the example:**  0.20 seconds
( 0 minutes  0.20 seconds)
    