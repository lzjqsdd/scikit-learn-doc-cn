.. _lda_qda:

==========================================
线性与二次判别分析法
==========================================

.. currentmodule:: sklearn

线性判别分析法(LDA)
(:class:`discriminant_analysis.LinearDiscriminantAnalysis`) 与二次判别分析法(QDA)(:class:`discriminant_analysis.QuadraticDiscriminantAnalysis`) 是两种经典的分类器，正如它们的名字所说，分别带有一个线性决策平面和一个二次决策平面。

这些分类器很吸引人，因为它们有可以容易计算求得闭式解，本质上是多类别分类，在实践中已被证明很有效，以及没有超参数需要调节。

.. |ldaqda| image:: ../auto_examples/classification/images/plot_lda_qda_001.png
        :target: ../auto_examples/classification/plot_lda_qda.html
        :scale: 80

.. centered:: |ldaqda|

上图展示了线性判决分析(LDA)和二次判决分析(QDA)的决策边界。底行的两图表明线性判决分析只能学习到线性边界，而二次判决分析可以学习到二次边界因此使用更加灵活。

.. topic:: Examples:

    :ref:`example_classification_plot_lda_qda.py` : LDA和QDA在模拟数据上的对比

使用LDA来降维
===========================================================

:class:`discriminant_analysis.LinearDiscriminantAnalysis` 可以被用来实现有监督的降维，通过把输入数据投影到一个由多个方向组成的线性子平面，那些方向可以最大化不同类别之间的间隔(下面用数学讨论会有更精确)。输出的维度必然会比类别的数目要小，因此这通常是一种较强的降维，也只在多类别分类中有意义。

:func:`discriminant_analysis.LinearDiscriminantAnalysis.transform` 是一种实现。可以使用构造器参数 ``n_components`` 来设置想要得到的维度。这个参数对
:func:`discriminant_analysis.LinearDiscriminantAnalysis.fit` 或者 :func:`discriminant_analysis.LinearDiscriminantAnalysis.predict` 没有影响。

.. topic:: Examples:

    :ref:`example_decomposition_plot_pca_vs_lda.py`: LDA和PCA降维在鸢尾花数据集上的对比

LDA和QDA分类器的数学表达
=======================================================

LDA和QDA都可以从简单的概率模型中的推导得到，概率模型可以对每一类别 :math:`k` 的类条件分别 :math:`P(X|y=k)` 进行建模。预测结果则可以通过贝叶斯法则来获得:

.. math::
    P(y=k | X) = \frac{P(X | y=k) P(y=k)}{P(X)} = \frac{P(X | y=k) P(y = k)}{ \sum_{l} P(X | y=l) \cdot P(y=l)}

于是我们选择可以最大化这个条件概率的类别 :math:`k` 。

更具体地说，对于线性和二次判别分析，:math:`P(X|y)` 为多变量的高斯分布，分布密度为:

.. math:: p(X | y=k) = \frac{1}{(2\pi)^n | \Sigma_k|^{1/2}}\exp\left(-\frac{1}{2} (X-\mu_k)^t \Sigma_k^{-1} (X-\mu_k)\right)

为了把这个模型当作分类器使用，我们只需要从训练数据中估计类先验概率 :math:`P(y=k)` (用类别 :math:`k` 的实例比例), 类别均值 :math:`\mu_k` (用经验的样本类别均值) 和协方差矩阵(通过用经验的样本类别协方差或者正则化的估计器:见下面的Shrinkage章节)。

对于LDA，对于每个类别的高斯分布假设共享相同的协方差矩阵:对于所有 :math:`k` 有 :math:`\Sigma_k = \Sigma` 。这可以带来线性的决策平面，正如所见通过比较log似然比 :math:`\log[P(y=k | X) / P(y=l | X)]` :

.. math::
   \log\left(\frac{P(y=k|X)}{P(y=l | X)}\right) = 0 \Leftrightarrow (\mu_k-\mu_l)\Sigma^{-1} X = \frac{1}{2} (\mu_k^t \Sigma^{-1} \mu_k - \mu_l^t \Sigma^{-1} \mu_l)

对于QDA，高斯分布的协方差矩阵 :math:`\Sigma_k` 则没有假设，因此能带来二次决策平面。更多细节见[#1]_ 。

.. note:: **和高斯朴素贝叶斯的关系**

	  如果在QDA模型中假设协方差矩阵是对角矩阵，这意味着我们假设类别之间是条件独立的，生成的分类器则等同于高斯朴素贝叶斯分类器 :class:`naive_bayes.GaussianNB` 。

LDA降维的数学表达
========================================================

为了理解LDA在降维中的使用，从上文介绍的LDA分类法则的几何推导讲起会比较有效。我们定义目标类别的总数目为 :math:`K` 。由于我们假设在LDA中所有的类别拥有相同的估计协方差 :math:`\Sigma` ，我们可以重缩放调整数据，使得协方差矩阵式恒等的:

.. math:: X^* = D^{-1/2}U^t X\text{ 使得 }\Sigma = UDU^t

因而可以看到，对一个调整后的数据点进行分类，等同于计算估计类别均值 :math:`\mu^*_k` ，这个均值在欧氏距离上最接近数据点。但这也可以在数据投影到 :math:`K-1` 个由所有类别的所有 :math:`\mu^*_k` 生成的仿射子空间 :math:`H_K` 之后完成。这表明在LDA分类器中，隐含着通过线性投影在一个 :math:`K-1` 维的空间来实现降维。

我们可以降低更低的维度到指定的 :math:`L` ，通过投影到可以最大化投影后的 :math:`\mu^*_k` 的方差的线性子空间 :math:`H_L` (事实上，我们正在对转换后的类别均值 :math:` \mu^*_k` 进行一种形式的PCA)。这个 :math:`L` 对应于:func:`discriminant_analysis.LinearDiscriminantAnalysis.transform` 方法中的 ``n_components`` 参数。更多细节见[#1]_ .

缩减技术(Shrinkage)
=====================

缩减(Shrinkage)是一个改善估计协方差矩阵的工具，用在当训练样本数小于特征数目的情况下。因为在这种场景中，根据经验来估计样本的协方差十分糟糕。Shrinkage LDA可以通过设置:class:`discriminant_analysis.LinearDiscriminantAnalysis` 类的参数 ``shrinkage`` 为'auto' 。通过Ledoit和Wolf [#2]_ 介绍的引理得到的分析方法，可以自动决定最优的shrinkage参数。值得注意的是，目前shrinkage只在设置 ``solver`` 参数为'lsqr'或者'eigen'时起作用。

``shrinkage`` 参数可以手动设置为0和1之间的取值。特别地，0的取值对应于没有shrinkage(这意味着将使用由经验得到的协方差矩阵)，1的取值对应于完全shrinkage(这意味着将会使用对角的方差矩阵来估计协方差矩阵)。设置这个参数为0和1之间的取值时，会得到一个协方差矩阵的shrunk版本的估计。 

.. |shrinkage| image:: ../auto_examples/classification/images/plot_lda_001.png
        :target: ../auto_examples/classification/plot_lda.html
        :scale: 75

.. centered:: |shrinkage|


估计算法
=====================

默认的解决方法是'svd'。它可以进行分类(classification)和转换(transform)，并且不依赖于协方差矩阵的计算。这对于当特征的数目很大时是个优点。然而，'svd'不能使用shrinkage。

'lsqr'算法是一种只用在分类中的有效算法，它支持shrinkage。

'eigen'算法基于类间离散度到类内离散率的优化，它可以用来分类和转换，同时支持shrinkage。但是，'eigen'算法需要计算协方差矩阵，因此它可能不适合于拥有大量特征的情形。

.. topic:: Examples:

    :ref: `example_classification_plot_lda.py` : LDA分类器有无shrinkage的对比。

.. topic:: References:

   .. [#1] "The Elements of Statistical Learning", Hastie T., Tibshirani R.,
      Friedman J., Section 4.3, p.106-119, 2008.

   .. [#2] Ledoit O, Wolf M. Honey, I Shrunk the Sample Covariance Matrix.
      The Journal of Portfolio Management 30(4), 110-119, 2004.
