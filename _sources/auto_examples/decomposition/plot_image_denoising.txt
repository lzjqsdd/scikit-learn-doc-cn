

.. _example_decomposition_plot_image_denoising.py:


=========================================
Image denoising using dictionary learning
=========================================

An example comparing the effect of reconstructing noisy fragments
of the Lena image using firstly online :ref:`DictionaryLearning` and
various transform methods.

The dictionary is fitted on the distorted left half of the image, and
subsequently used to reconstruct the right half. Note that even better
performance could be achieved by fitting to an undistorted (i.e.
noiseless) image, but here we start from the assumption that it is not
available.

A common practice for evaluating the results of image denoising is by looking
at the difference between the reconstruction and the original image. If the
reconstruction is perfect this will look like Gaussian noise.

It can be seen from the plots that the results of :ref:`omp` with two
non-zero coefficients is a bit less biased than when keeping only one
(the edges look less prominent). It is in addition closer from the ground
truth in Frobenius norm.

The result of :ref:`least_angle_regression` is much more strongly biased: the
difference is reminiscent of the local intensity value of the original image.

Thresholding is clearly not useful for denoising, but it is here to show that
it can produce a suggestive output with very high speed, and thus be useful
for other tasks such as object classification, where performance is not
necessarily related to visualisation.




.. rst-class:: horizontal





**Python source code:** :download:`plot_image_denoising.py <plot_image_denoising.py>`

.. literalinclude:: plot_image_denoising.py
    :lines: 34-

**Total running time of the example:**  0.00 seconds
( 0 minutes  0.00 seconds)
    