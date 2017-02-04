.. currentmodule:: sklearn.grid_search

.. _grid_search:

===============================================
网格搜索: 找到最好的估计器
===============================================

Parameters that are not directly learnt within estimators can be set by
searching a parameter space for the best :ref:`cross_validation` score.
Typical examples include ``C``, ``kernel`` and ``gamma`` for Support Vector
Classifier, ``alpha`` for Lasso, etc. (对于估计器不能够直接学习的参数，可以设定一个搜索空间，
找到能够取得最高分数 :ref:`cross_validation` 的参数。典型的例子有支持向量机分类器中的
 ``C``, ``kernel`` 和 ``gamma``， Lasso 中的 ``alpha`` 等。)

Any parameter provided when constructing an estimator may be optimized in this
manner.  Specifically, to find the names and current values for all parameters
for a given estimator, use (当构建一个估计器时，任何参数都可以使用这种方式来优化。特别的，
使用下面的方法来得到给定估计器所有参数的名称和当前的值)::

  estimator.get_params()

Such parameters are often referred to as *hyperparameters* (particularly in
Bayesian learning), distinguishing them from the parameters optimised in a
machine learning procedure (这些参数通常称之为 *超参数* (特别是在 Bayesian 学习中)，这使得他们和
机器学习过程中红优化的参数区别开来).

A search consists of(一个搜索包括):

- an estimator (regressor or classifier such as ``sklearn.svm.SVC()``) 一个估计器 (回归或者分类器，例如 ``sklearn.svm.SVC()``);
- a parameter space (参数空间);
- a method for searching or sampling candidates (一个用于搜索和采样候选值的方法);
- a cross-validation scheme (交叉验证策略); and
- a :ref:`score function <gridsearch_scoring>` (一个分数计算函数 :ref:`score function <gridsearch_scoring>` ).

Some models allow for specialized, efficient parameter search strategies,
:ref:`outlined below <alternative_cv>`.
Two generic approaches to sampling search candidates are provided in
scikit-learn: for given values, :class:`GridSearchCV` exhaustively considers
all parameter combinations, while :class:`RandomizedSearchCV` can sample a
given number of candidates from a parameter space with a specified
distribution. After describing these tools we detail
:ref:`best practice <grid_search_tips>` applicable to both approaches. (一些模型允许使用特别的，
高效的参数搜索策略，:ref:`outlined below <alternative_cv>`。在 scikit-learn 中给出了两种通用的用于
搜索候选区间的方法：对于给定的值， :class:`GridSearchCV` 会搜索全部的参数组合，而 :class:`RandomizedSearchCV` 
会从参数空间中使用特定的分布给出一些参数的候选用于搜索。在描述这些工具之后，:ref:`best practice <grid_search_tips>` 
详细讲解如何用用这两种方法。)

穷尽网格搜索(Exhaustive Grid Search)
=================================

The grid search provided by :class:`GridSearchCV` exhaustively generates
candidates from a grid of parameter values specified with the ``param_grid``
parameter. For instance, the following ``param_grid`` (由 :class:`GridSearchCV` 给出的
网格搜索方法会穷尽搜索由 ``param_grid`` 参数设定网格中所有的候选项)::

  param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
   ]

specifies that two grids should be explored: one with a linear kernel and
C values in [1, 10, 100, 1000], and the second one with an RBF kernel,
and the cross-product of C values ranging in [1, 10, 100, 1000] and gamma
values in [0.001, 0.0001]. (特别的，两个网格会被搜索：一个拥有线性核而且 C 的值在 [1, 10, 100, 1000]，
而另一个拥有 RBF 核，C 的值在 [1, 10, 100, 1000] 而 gamma 的取值在 [0.001, 0.0001] 中)。

The :class:`GridSearchCV` instance implements the usual estimator API: when
"fitting" it on a dataset all the possible combinations of parameter values are
evaluated and the best combination is retained. ( :class:`GridSearchCV` 实例实现了通常的估计器 API：
当使用其在某个数据集上进行拟合，评估所有可能的参数值的组合结果时，能够获得最好的参数组合。)

.. currentmodule:: sklearn.grid_search

.. topic:: Examples (例子):

    - See :ref:`example_model_selection_grid_search_digits.py` for an example of
      Grid Search computation on the digits dataset. (查看 :ref:`example_model_selection_grid_search_digits.py`
      是一个在数字数据集上进行网格搜索的例子。)

    - See :ref:`example_model_selection_grid_search_text_feature_extraction.py` for an example
      of Grid Search coupling parameters from a text documents feature
      extractor (n-gram count vectorizer and TF-IDF transformer) with a
      classifier (here a linear SVM trained with SGD with either elastic
      net or L2 penalty) using a :class:`pipeline.Pipeline` instance. (查看 :ref:`example_model_selection_grid_search_text_feature_extraction.py`
      是一个在文本特征提取器上使用网格搜索的例子 (n-gram 计数向量化和 TF_IDF 转换) 和一个分类器 (一个使用 SGD 训练、使用 elastic net 或者 L2 损失函数的线性 SVM)，
      使用 :class:`pipeline.Pipeline` 的实例)。

.. _randomized_parameter_search:

Randomized Parameter Optimization (随机参数优化)
==============================================
While using a grid of parameter settings is currently the most widely used
method for parameter optimization, other search methods have more
favourable properties. (使用一个网格的参数设置并搜索
是当前常用的参数优化手段，然而一些其他的搜索方法也有一些自己特有的性质)
:class:`RandomizedSearchCV` implements a randomized search over parameters,
where each setting is sampled from a distribution over possible parameter values.
This has two main benefits over an exhaustive search (:class:`RandomizedSearchCV` 实现了一个在参数上随机搜索的方法，
每次使用的参数是使用一个特别的分布从允许的参数列表中选择。相比于群举搜索，有以下好处):

* A budget can be chosen independent of the number of parameters and possible values. (可以独立地控制计算代价，与参数空间的选择无关)
* Adding parameters that do not influence the performance does not decrease efficiency. (增加参数但不会影响结果或者降低性能)

Specifying how parameters should be sampled is done using a dictionary, very
similar to specifying parameters for :class:`GridSearchCV`. Additionally,
a computation budget, being the number of sampled candidates or sampling
iterations, is specified using the ``n_iter`` parameter.
For each parameter, either a distribution over possible values or a list of
discrete choices (which will be sampled uniformly) can be specified (使用字典来设定如何进行参数采样，
和使用 :class:`GridSearchCV` 非常类似。此外，使用 ``n_iter`` 参数可以通过设定候选采样参数的数量来设定计算预算。
对于这些参数，或者是允许值的分布或者是离散值的列表（将会按照均一概率采样）来进行设定)::

  [{'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1),
    'kernel': ['rbf'], 'class_weight':['auto', None]}]

This example uses the ``scipy.stats`` module, which contains many useful
distributions for sampling parameters, such as ``expon``, ``gamma``,
``uniform`` or ``randint``. (这个例子使用了 ``scipy.stats`` 模块，这个模块包括了许多可以用于参数采样的分布，如
``expon``, ``gamma``, ``uniform`` 或者 ``randint``)
In principle, any function can be passed that provides a ``rvs`` (random
variate sample) method to sample a value. A call to the ``rvs`` function should
provide independent random samples from possible parameter values on
consecutive calls. (实际上，给任意函数传入一个 ``rvs`` (随机变量采样) 方法来采样获得一个值。调用 ``rvs`` 的函数在
连续调用时，应该提供独立的随机采样)

    .. warning::

        The distributions in ``scipy.stats`` do not allow specifying a random
        state. Instead, they use the global numpy random state, that can be seeded
        via ``np.random.seed`` or set using ``np.random.set_state``. (在 ``scipy.stats`` 中并不允许设定特有的随机状态，
        而是使用全局的 numpy 的随机状态，可以使用 ``np.random.seed`` 获得对应的 seed，或者使用 ``np.random.set_state`` 来设定。)

For continuous parameters, such as ``C`` above, it is important to specify
a continuous distribution to take full advantage of the randomization. This way,
increasing ``n_iter`` will always lead to a finer search. (对于连续的参数，比如上面的 ``C``，设定一个连续的分布使得随机性的全部好处发挥出来十分重要。
这种方式下，增加 ``n_iter`` 总会得到一个更好的搜索结果。) 

.. topic:: Examples:

    * :ref:`example_model_selection_randomized_search.py` compares the usage and efficiency
      of randomized search and grid search. (:ref:`example_model_selection_randomized_search.py` 比较了
      随机搜索和网格搜索的使用和效率。)

.. topic:: References:

    * Bergstra, J. and Bengio, Y.,
      Random search for hyper-parameter optimization,
      The Journal of Machine Learning Research (2012)

.. _grid_search_tips:

Tips for parameter search (参数搜索的技巧)
=========================================

.. _gridsearch_scoring:

Specifying an objective metric (设定一个评估标准)
----------------------------------------------

By default, parameter search uses the ``score`` function of the estimator
to evaluate a parameter setting. These are the
:func:`sklearn.metrics.accuracy_score` for classification and
:func:`sklearn.metrics.r2_score` for regression.  For some applications,
other scoring functions are better suited (for example in unbalanced
classification, the accuracy score is often uninformative). An alternative
scoring function can be specified via the ``scoring`` parameter to
:class:`GridSearchCV`, :class:`RandomizedSearchCV` and many of the
specialized cross-validation tools described below.
See :ref:`scoring_parameter` for more details.
默认地，参数搜索使用 ``score`` 函数来评估估计器参数的取值效果。已经有用于分类的 :func:`sklearn.metrics.accuracy_score`
和用于回归的 :func:`sklearn.metrics.r2_score`。一些其他的应用情况下，可能需要特有的函数来计算分数 (例如：在不均衡分类中，
精确度分数通常信息含量不高。) 可以通过给参数 ``scoring`` 设定特有的计算函数来定制 :class:`GridSearchCV`, :class:`RandomizedSearchCV`以及
大部分上面提到的交叉验证工具。阅读 :ref:`scoring_parameter` 可以得到更多的细节。


Composite estimators and parameter spaces (组合评估器和参数空间)
-------------------------------------------------------------

:ref:`pipeline` describes building composite estimators whose
parameter space can be searched with these tools. 
:ref:`pipeline` 描述了构建组合参数估计器以及在组合参数空间中进行搜索的工具。

Model selection: development and evaluation (模型选择：开发和评估)
---------------------------------------------------------------

Model selection by evaluating various parameter settings can be seen as a way
to use the labeled data to "train" the parameters of the grid.
通过评估不同的参数设定来选择模型可以看做是一种带标签的数据来训练 grid 参数的过程。

When evaluating the resulting model it is important to do it on
held-out samples that were not seen during the grid search process:
it is recommended to split the data into a **development set** (to
be fed to the ``GridSearchCV`` instance) and an **evaluation set**
to compute performance metrics.
当进行评估结果模型时，保留一部分样例不在搜索时使用，单独用于评估是非常重要的手段：
建议将数据划分为 **开发集** (``GridSearchCV`` 实例使用) 和 **评估集** 来衡量结果。

This can be done by using the :func:`cross_validation.train_test_split`
utility function. 这个可以通过 :func:`cross_validation.train_test_split` 函数实现。

Parallelism (并行化)
-------------------

:class:`GridSearchCV` and :class:`RandomizedSearchCV` evaluate each parameter
setting independently.  Computations can be run in parallel if your OS
supports it, by using the keyword ``n_jobs=-1``. See function signature for
more details.
:class:`GridSearchCV` 和 :class:`RandomizedSearchCV` 独立地评估每个参数。如果使用的 OS 允许，
计算可以并行地进行，设定参数 ``n_jobs=-1`` 来支持这一特性。阅读函数签名来获得更加详细的信息。

Robustness to failure (失败的鲁棒性)
----------------------------------

Some parameter settings may result in a failure to ``fit`` one or more folds
of the data.  By default, this will cause the entire search to fail, even if
some parameter settings could be fully evaluated. Setting ``error_score=0``
(or `=np.NaN`) will make the procedure robust to such failure, issuing a
warning and setting the score for that fold to 0 (or `NaN`), but completing
the search.
一些参数的设定可能会引起一个或者多个 fold 拟合的失败。默认地，这将会引起整个搜索的失败，即使
一些参数设定能够被完全求值。设定 ``error_score=0`` (或者 `=np.NaN`) 将会使得过程遇到这种失败时
仍然保持鲁棒性，这会引起 warning，然后设定这个 fold 的分数为 0 或者 `NaN`，但是最终搜索会完成。

.. _alternative_cv:

Alternatives to brute force parameter search (暴力参数搜索的替代)
===============================================================

Model specific cross-validation (模型特定的交叉验证)
-------------------------------------------------


Some models can fit data for a range of value of some parameter almost
as efficiently as fitting the estimator for a single value of the
parameter. This feature can be leveraged to perform a more efficient
cross-validation used for model selection of this parameter. (一些模型在一些参数取值为某个范围
内训练时其效率和仅训练某单个参数时大致相同。这个特征可以用于提升交叉验证的效率用于基于交叉验证的模型选择)

The most common parameter amenable to this strategy is the parameter
encoding the strength of the regularizer. In this case we say that we
compute the **regularization path** of the estimator. (这种策略常用于正则参数的编码强化。
这种情况下，计算估计器的 **regularization path**)

Here is the list of such models (这种模型的列表如下):

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   linear_model.ElasticNetCV
   linear_model.LarsCV
   linear_model.LassoCV
   linear_model.LassoLarsCV
   linear_model.LogisticRegressionCV
   linear_model.MultiTaskElasticNetCV
   linear_model.MultiTaskLassoCV
   linear_model.OrthogonalMatchingPursuitCV
   linear_model.RidgeCV
   linear_model.RidgeClassifierCV


Information Criterion (信息规范)
-------------------------------

Some models can offer an information-theoretic closed-form formula of the
optimal estimate of the regularization parameter by computing a single
regularization path (instead of several when using cross-validation). (一些模型通过计算规则路径
（而不是使用交叉验证) 可以给出提供在信息理论上可解析的最优规则参数计算公式。

Here is the list of models benefitting from the Aikike Information
Criterion (AIC) or the Bayesian Information Criterion (BIC) for automated
model selection (这是一个可从 AIC 信息准则或者 BIC 信息准则中获益的模型列表):

.. autosummary::
   :toctree: generated/
   :template: class.rst

   linear_model.LassoLarsIC


.. _out_of_bag:

Out of Bag Estimates 实用化的评估
-------------------------------

When using ensemble methods base upon bagging, i.e. generating new
training sets using sampling with replacement, part of the training set
remains unused.  For each classifier in the ensemble, a different part
of the training set is left out. (当使用提升方法比如 bagging, 当使用可替代采样重新生成训练集合时，有一部分训练数据集仍然没有使用，
对于提升集合中的每个分类器，一个不同的训练集合被保留出来)

This left out portion can be used to estimate the generalization error
without having to rely on a separate validation set.  This estimate
comes "for free" as no additional data is needed and can be used for
model selection. 留出来的部分可以用于评估泛化误差，而不需要另外依赖于一个单独的验证集合。
这个估计不需要额外的数据，可以用于模型选择。

This is currently implemented in the following classes 在下面的类中实现了这个方法:

.. autosummary::
   :toctree: generated/
   :template: class.rst

    ensemble.RandomForestClassifier
    ensemble.RandomForestRegressor
    ensemble.ExtraTreesClassifier
    ensemble.ExtraTreesRegressor
    ensemble.GradientBoostingClassifier
    ensemble.GradientBoostingRegressor
