.. _isotonic:

===================
保序回归(Isotonic regression)
===================

.. currentmodule:: sklearn.isotonic

类 :class:`IsotonicRegression` 用数据拟合一个非递减逼近函数.
解决了如下问题:

  使 :math:`\sum_i w_i (y_i - \hat{y}_i)^2`　最小化

  约束条件： :math:`\hat{y}_{min} = \hat{y}_1 \le \hat{y}_2 ... \le \hat{y}_n = \hat{y}_{max}`

其中 :math:`w_i` 严格大于０， :math:`y_i` 是任意实数．它返回一个由非递减元素组成的向量．这些元素根据均方误差最小化得到．事实上，这个元素序列组成一个分段线性函数．

.. figure:: ../auto_examples/images/plot_isotonic_regression_001.png
   :target: ../auto_examples/images/plot_isotonic_regression.html
   :align: center
