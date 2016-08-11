

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

一个开场的回归样例
----------------------------------

Say we want to surrogate the function :math:`g(x) = x \sin(x)`. To do so,
the function is evaluated onto a design of experiments. Then, we define a
GaussianProcess model whose regression and correlation models might be
specified using additional kwargs, and ask for the model to be fitted to the
data. Depending on the number of parameters provided at instantiation, the
fitting procedure may recourse to maximum likelihood estimation for the
parameters or alternatively it uses the given parameters.

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


Fitting Noisy Data
------------------

When the data to be fit includes noise, the Gaussian process model can be
used by specifying the variance of the noise for each point.
:class:`GaussianProcess` takes a parameter ``nugget`` which
is added to the diagonal of the correlation matrix between training points:
in general this is a type of Tikhonov regularization.  In the special case
of a squared-exponential correlation function, this normalization is
equivalent to specifying a fractional variance in the input.  That is

.. math::
   \mathrm{nugget}_i = \left[\frac{\sigma_i}{y_i}\right]^2

With ``nugget`` and ``corr`` properly set, Gaussian Processes can be
used to robustly recover an underlying function from noisy data:

.. figure:: ../auto_examples/gaussian_process/images/plot_gp_regression_002.png
   :target: ../auto_examples/gaussian_process/plot_gp_regression.html
   :align: center

.. topic:: Other examples

  * :ref:`example_gaussian_process_plot_gp_probabilistic_classification_after_regression.py`



Mathematical formulation
========================


The initial assumption
----------------------

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
