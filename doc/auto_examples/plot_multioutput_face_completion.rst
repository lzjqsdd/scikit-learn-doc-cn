

.. _example_plot_multioutput_face_completion.py:


==============================================
Face completion with a multi-output estimators
==============================================

This example shows the use of multi-output estimator to complete images.
The goal is to predict the lower half of a face given its upper half.

The first column of images shows true faces. The next columns illustrate
how extremely randomized trees, k nearest neighbors, linear
regression and ridge regression complete the lower half of those faces.




.. image:: images/plot_multioutput_face_completion_001.png
    :align: center




**Python source code:** :download:`plot_multioutput_face_completion.py <plot_multioutput_face_completion.py>`

.. literalinclude:: plot_multioutput_face_completion.py
    :lines: 14-

**Total running time of the example:**  7.46 seconds
( 0 minutes  7.46 seconds)
    