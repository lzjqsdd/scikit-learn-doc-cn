.. _semi_supervised:

===================================================
半监督
===================================================

.. currentmodule:: sklearn.semi_supervised

`半监督学习
<http://en.wikipedia.org/wiki/Semi-supervised_learning>`_ 指的是在你的训练数据当中，一些样本没有被标注的情况. 半监督的评价者in :mod:`sklearn.semi_supervised` 可以利用这些额外的未标注的数据更好地捕捉到潜在的数据分布形式，并且能够更好的泛化到新样本上.
当我们拥有少量标记数据以及大量的未标记数据的时候，这些算法能够取得很好的表现.

.. topic:: Unlabeled entries in `y`

	在使用``fit``方法训练模型时，必须为未标记的点以及标记的数据分配标识符。 此实现使用的标识符是整数值:math:`-1`.

.. _label_propagation:

标签传播算法
=================

标签传播算法时半监督图推论算法的一个变种形式. 

这种模型的特点:
  * 可以用于分类和回归任务
  * Kernel methods 将数据投射到 alternate dimensional spaces

`scikit-learn` 提供了两个标签传播模型:
:class:`LabelPropagation` and :class:`LabelSpreading`. 两者都通过在输入数据集中的所有项目上构造相似性图来工作。

.. figure:: ../auto_examples/semi_supervised/images/plot_label_propagation_structure_001.png
    :target: ../auto_examples/semi_supervised/plot_label_propagation_structure.html
    :align: center
    :scale: 60%

    **一个标签传播算法的示例:** *无标记观测结构与类结构一致，因此类标签可以传播给训练集的无标签观测。*

:class:`LabelPropagation` 和 :class:`LabelSpreading`
的不同之处在于，对图的相似度矩阵和对标签分布的夹逼效应的修改上。
夹逼效应允许算法在一定程度上修改真实标记的权重。:class:`LabelPropagation` algorithm performs hard
clamping of input labels, 这意味着 :math:`\alpha=1`。 夹逼因子可以是随意的，比如:math:`\alpha=0.8`, 意思是我们将始终保留我们的原始标签分布的80%，
但算法可以在20％范围内改变其分布的置信度。

:class:`LabelPropagation`使用由未经修改的数据构造的原始相似度矩阵。 相反地，
:class:`LabelSpreading`则利用最小化一个包含正则项的损失函数的方式，这样做通常对于噪声更加鲁棒。
算法在原始图的修改版本上迭代，并通过计算归一化图的拉普拉斯矩阵来归一化边缘权重。这个过程也用于:ref:`spectral_clustering`。

标签传播模型有两个内置的核函数（kernel method），不同的核函数能够影响算法的scalability和performance。如下：

  * rbf (:math:`\exp(-\gamma |x-y|^2), \gamma > 0`). :math:`\gamma` is
    specified by keyword gamma.

  * knn (:math:`1[x' \in kNN(x)]`). :math:`k` is specified by keyword
    n_neighbors.

The RBF kernel（径向基核函数）将产生一个全连接图，这个全连接图在内存中以一个稠密矩阵的形式表示。
该矩阵可能非常大，并且当它和算法的每次迭代执行的完整矩阵相乘运算的计算成本结合在一起的时候，可能导致过长的运行时间。
另一种情况,也就是 KNN kernel 生成一个足够对内存友好的稀疏矩阵，这将极大的削减运行时间。

.. topic:: 示例

  * :ref:`example_semi_supervised_plot_label_propagation_versus_svm_iris.py`
  * :ref:`example_semi_supervised_plot_label_propagation_structure.py`
  * :ref:`example_semi_supervised_plot_label_propagation_digits_active_learning.py`

.. topic:: 参考

    [1] Yoshua Bengio, Olivier Delalleau, Nicolas Le Roux. In Semi-Supervised
    Learning (2006), pp. 193-216

    [2] Olivier Delalleau, Yoshua Bengio, Nicolas Le Roux. Efficient
    Non-Parametric Function Induction in Semi-Supervised Learning. AISTAT 2005
    http://research.microsoft.com/en-us/people/nicolasl/efficient_ssl.pdf

