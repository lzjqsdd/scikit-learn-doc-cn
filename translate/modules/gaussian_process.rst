

.. _gaussian_process:

=============================
高斯过程(Gaussian Processes)
=============================

.. currentmodule:: sklearn.gaussian_process

**针对机器学习的高斯过程(Gaussian Processes for Machine Learning,即 GPML)** 是一个通用的监督学习方法，主要被设计用来解决 *回归* 问题。
它也可以扩展为 *概率分类(probabilistic classification)*，但是在当前的实现中，这只是 *回归* 练习的一个后续处理。

GPML的优势如下:

    - 预测是对观察值的插值（至少在普通相关模型上是的）.

    - 预测是带有概率的(Gaussian)。所以可以用来计算经验置信区间和超越概率
      以便对感兴趣的区域重新拟合（在线拟合，自适应拟合）预测。

    - 多样性: 可以指定不同的线性回归模型 :ref:`linear regression models <linear_model>`
      和相关模型 :ref:`correlation models <correlation_models>` 。
      它提供了普通模型，但也能指定其它静态的自定义模型

GPML的缺点如下:

    - 不是稀疏的，它使用全部的样本/特征信息来做预测。

    - 多维空间下会变得低效 -- 即当特征的数量超过几十个,它可能确实会表现很差，而且计算效率下降。

    - 分类只是一个后处理过程, 意味着要建模，
      首先需要提供试验的完整浮点精度标量输出 :math:`y` 来解决回归问题。

要感谢高斯的预测的属性，已经有了广泛应用，比如：最优化和概率分类

Examples
========

用一个回归样例来开场
----------------------------------

比如说，我们要代替函数:math:`g(x) = x \sin(x)`。首先，要在一系列设计好的试验上
对这个函数求值。
然后，我们定义了一个GaussianProcess模型，它的回归模型和相关模型可能会通过附加的kwargs来指明，并调用模型来拟合数据。
根据实例提供的参数的数量，拟合程序可能依靠参数的最大似然估计或者是就使用给定的参数本身。

.. figure:: ../auto_examples/gaussian_process/images/plot_gp_regression_001.png
   :target: ../auto_examples/gaussian_process/plot_gp_regression.html
   :align: center

::

    >>> import numpy as np
    >>> from sklearn import gaussian_process
    >>> def f(x):
    ...	    return x * np.sin(x)
    >>> X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
    >>> y = f(X).ravel()
    >>> x = np.atleast_2d(np.linspace(0, 10, 1000)).T
    >>> gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
    >>> gp.fit(X, y)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    GaussianProcess(beta0=None, corr=<function squared_exponential at 0x...>,
            normalize=True, nugget=array(2.22...-15),
            optimizer='fmin_cobyla', random_start=1, random_state=...
            regr=<function constant at 0x...>, storage_mode='full',
            theta0=array([[ 0.01]]), thetaL=array([[ 0.0001]]),
            thetaU=array([[ 0.1]]), verbose=False)
    >>> y_pred, sigma2_pred = gp.predict(x, eval_MSE=True)


拟合噪声数据
------------------

当要拟合的数据有噪声时，高斯过程模型能够通过用指定每个点的噪声方差来使用。

:class:`GaussianProcess` 接收一个 ``nugget`` 参数，这个参数会被加到训练数据相关矩阵的对角线上:
一般来说，这是Tikhonov正则化 的其中一种类型。 在平方指数(squared-exponential SE)相关函数的特殊情形下，这个正则相当于是指定了输入的误差方差。
即

.. math::
   \mathrm{nugget}_i = \left[\frac{\sigma_i}{y_i}\right]^2

在 ``nugget`` 和 ``corr`` 设置合适的情况下，高斯过程可以鲁棒地从噪声数据中恢复出一个基本函数:

.. figure:: ../auto_examples/gaussian_process/images/plot_gp_regression_002.png
   :target: ../auto_examples/gaussian_process/plot_gp_regression.html
   :align: center

.. topic:: 其它样例

  * :ref:`example_gaussian_process_plot_gp_probabilistic_classification_after_regression.py`



数学 公式
===========


初始假设
---------

Suppose one wants to model the output of a computer experiment, say a
mathematical function:

.. math::

        g: & \mathbb{R}^{n_{\rm features}} \rightarrow \mathbb{R} \\
           & X \mapsto y = g(X)

GPML starts with the assumption that this function is *a* conditional sample
path of *a* Gaussian process :math:`G` which is additionally assumed to read as
follows:

.. math::

        G(X) = f(X)^T \beta + Z(X)

where :math:`f(X)^T \beta` is a linear regression model and :math:`Z(X)` is a
zero-mean Gaussian process with a fully stationary covariance function:

.. math::

        C(X, X') = \sigma^2 R(|X - X'|)

:math:`\sigma^2` being its variance and :math:`R` being the correlation
function which solely depends on the absolute relative distance between each
sample, possibly featurewise (this is the stationarity assumption).

From this basic formulation, note that GPML is nothing but an extension of a
basic least squares linear regression problem:

.. math::

        g(X) \approx f(X)^T \beta

Except we additionally assume some spatial coherence (correlation) between the
samples dictated by the correlation function. Indeed, ordinary least squares
assumes the correlation model :math:`R(|X - X'|)` is one when :math:`X = X'`
and zero otherwise : a *dirac* correlation model -- sometimes referred to as a
*nugget* correlation model in the kriging literature.


The best linear unbiased prediction (BLUP)
------------------------------------------

We now derive the *best linear unbiased prediction* of the sample path
:math:`g` conditioned on the observations:

.. math::

    \hat{G}(X) = G(X | y_1 = g(X_1), ...,
                                y_{n_{\rm samples}} = g(X_{n_{\rm samples}}))

It is derived from its *given properties*:

- It is linear (a linear combination of the observations)

.. math::

    \hat{G}(X) \equiv a(X)^T y

- It is unbiased

.. math::

    \mathbb{E}[G(X) - \hat{G}(X)] = 0

- It is the best (in the Mean Squared Error sense)

.. math::

    \hat{G}(X)^* = \arg \min\limits_{\hat{G}(X)} \;
                                            \mathbb{E}[(G(X) - \hat{G}(X))^2]

So that the optimal weight vector :math:`a(X)` is solution of the following
equality constrained optimization problem:

.. math::

    a(X)^* = \arg \min\limits_{a(X)} & \; \mathbb{E}[(G(X) - a(X)^T y)^2] \\
                       {\rm s. t.} & \; \mathbb{E}[G(X) - a(X)^T y] = 0

Rewriting this constrained optimization problem in the form of a Lagrangian and
looking further for the first order optimality conditions to be satisfied, one
ends up with a closed form expression for the sought predictor -- see
references for the complete proof.

In the end, the BLUP is shown to be a Gaussian random variate with mean:

.. math::

    \mu_{\hat{Y}}(X) = f(X)^T\,\hat{\beta} + r(X)^T\,\gamma

and variance:

.. math::

    \sigma_{\hat{Y}}^2(X) = \sigma_{Y}^2\,
    ( 1
    - r(X)^T\,R^{-1}\,r(X)
    + u(X)^T\,(F^T\,R^{-1}\,F)^{-1}\,u(X)
    )

where we have introduced:

* the correlation matrix whose terms are defined wrt the autocorrelation
  function and its built-in parameters :math:`\theta`:

.. math::

    R_{i\,j} = R(|X_i - X_j|, \theta), \; i,\,j = 1, ..., m

* the vector of cross-correlations between the point where the prediction is
  made and the points in the DOE:

.. math::

    r_i = R(|X - X_i|, \theta), \; i = 1, ..., m

* the regression matrix (eg the Vandermonde matrix if :math:`f` is a polynomial
  basis):

.. math::

    F_{i\,j} = f_i(X_j), \; i = 1, ..., p, \, j = 1, ..., m

* the generalized least square regression weights:

.. math::

    \hat{\beta} =(F^T\,R^{-1}\,F)^{-1}\,F^T\,R^{-1}\,Y

* and the vectors:

.. math::

    \gamma & = R^{-1}(Y - F\,\hat{\beta}) \\
    u(X) & = F^T\,R^{-1}\,r(X) - f(X)

It is important to notice that the probabilistic response of a Gaussian Process
predictor is fully analytic and mostly relies on basic linear algebra
operations. More precisely the mean prediction is the sum of two simple linear
combinations (dot products), and the variance requires two matrix inversions,
but the correlation matrix can be decomposed only once using a Cholesky
decomposition algorithm.


The empirical best linear unbiased predictor (EBLUP)
----------------------------------------------------

Until now, both the autocorrelation and regression models were assumed given.
In practice however they are never known in advance so that one has to make
(motivated) empirical choices for these models :ref:`correlation_models`.

Provided these choices are made, one should estimate the remaining unknown
parameters involved in the BLUP. To do so, one uses the set of provided
observations in conjunction with some inference technique. The present
implementation, which is based on the DACE's Matlab toolbox uses the *maximum
likelihood estimation* technique -- see DACE manual in references for the
complete equations. This maximum likelihood estimation problem is turned into
a global optimization problem onto the autocorrelation parameters. In the
present implementation, this global optimization is solved by means of the
fmin_cobyla optimization function from scipy.optimize. In the case of
anisotropy however, we provide an implementation of Welch's componentwise
optimization algorithm -- see references.

For a more comprehensive description of the theoretical aspects of Gaussian
Processes for Machine Learning, please refer to the references below:

.. topic:: References:

    * `DACE, A Matlab Kriging Toolbox
      <http://www2.imm.dtu.dk/~hbn/dace/>`_ S Lophaven, HB Nielsen, J
      Sondergaard 2002


    * `Screening, predicting, and computer experiments
      <http://www.jstor.org/pss/1269548>`_ WJ Welch, RJ Buck, J Sacks,
      HP Wynn, TJ Mitchell, and MD Morris Technometrics 34(1) 15--25,
      1992


    * `Gaussian Processes for Machine Learning
      <http://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_ CE
      Rasmussen, CKI Williams MIT Press, 2006 (Ed. T Diettrich)


    * `The design and analysis of computer experiments
      <http://www.stat.osu.edu/~comp_exp/book.html>`_ TJ Santner, BJ
      Williams, W Notz Springer, 2003



.. _correlation_models:

Correlation Models
==================

Common correlation models matches some famous SVM's kernels because they are
mostly built on equivalent assumptions. They must fulfill Mercer's conditions
and should additionally remain stationary. Note however, that the choice of the
correlation model should be made in agreement with the known properties of the
original experiment from which the observations come. For instance:

* If the original experiment is known to be infinitely differentiable (smooth),
  then one should use the *squared-exponential correlation model*.
* If it's not, then one should rather use the *exponential correlation model*.
* Note also that there exists a correlation model that takes the degree of
  derivability as input: this is the Matern correlation model, but it's not
  implemented here (TODO).

For a more detailed discussion on the selection of appropriate correlation
models, see the book by Rasmussen & Williams in references.

.. _regression_models:


Regression Models
=================

Common linear regression models involve zero- (constant), first- and
second-order polynomials. But one may specify its own in the form of a Python
function that takes the features X as input and that returns a vector
containing the values of the functional set. The only constraint is that the
number of functions must not exceed the number of available observations so
that the underlying regression problem is not *underdetermined*.


Implementation details
======================

The present implementation is based on a translation of the DACE Matlab
toolbox.

.. topic:: References:

    * `DACE, A Matlab Kriging Toolbox
      <http://www2.imm.dtu.dk/~hbn/dace/>`_ S Lophaven, HB Nielsen, J
      Sondergaard 2002,

    * W.J. Welch, R.J. Buck, J. Sacks, H.P. Wynn, T.J. Mitchell, and M.D.
      Morris (1992). Screening, predicting, and computer experiments.
      Technometrics, 34(1) 15--25.
