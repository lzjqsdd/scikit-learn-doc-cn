.. _cross_decomposition:

===================
互分解(Cross decomposition)
===================

.. currentmodule:: sklearn.cross_decomposition

互分解模块主要包含如下两类算法：偏最小二乘法（PLS）和典型相关分析（CCA)

这一类算法主要用来寻找两个多元数据集的线性关系，fit函数中的参数X,Y都是2维数组

.. figure:: ../auto_examples/cross_decomposition/images/plot_compare_cross_decomposition_001.png
   :target: ../auto_examples/cross_decomposition/plot_compare_cross_decomposition.html
   :scale: 75%
   :align: center

互分解算法用于查找两个矩阵（X和Y）的基本关系，是一个在X,Y这两个空间对协方差结构建模的隐变量方法。它试图找到X空间的多维方向来解释Y空间方差最大的多维方向。偏最小二乘回归特别适合当预测矩阵比观测矩阵有更多变量，以及X的值中有多重共线性的时候。相比之下，标准的回归在这些情况下是不见效的。

这个模块主要包含如下几个类 :class:`PLSRegression`
:class:`PLSCanonical`, :class:`CCA` and :class:`PLSSVD`


.. topic:: 参考文献:

   * JA Wegelin
     `A survey of Partial Least Squares (PLS) methods, with emphasis on the two-block case <https://www.stat.washington.edu/research/reports/2000/tr371.pdf>`_

.. topic:: 示例:

    * :ref:`example_cross_decomposition_plot_compare_cross_decomposition.py`
