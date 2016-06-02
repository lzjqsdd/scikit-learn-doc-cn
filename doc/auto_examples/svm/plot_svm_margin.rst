

.. _example_svm_plot_svm_margin.py:


=========================================================
SVM Margins Example
=========================================================
The plots below illustrate the effect the parameter `C` has
on the separation line. A large value of `C` basically tells
our model that we do not have that much faith in our data's
distribution, and will only consider points close to line
of separation.

A small value of `C` includes more/all the observations, allowing
the margins to be calculated using all the data in the area.




.. rst-class:: horizontal


    *

      .. image:: images/plot_svm_margin_001.png
            :scale: 47

    *

      .. image:: images/plot_svm_margin_002.png
            :scale: 47




**Python source code:** :download:`plot_svm_margin.py <plot_svm_margin.py>`

.. literalinclude:: plot_svm_margin.py
    :lines: 18-

**Total running time of the example:**  0.13 seconds
( 0 minutes  0.13 seconds)
    