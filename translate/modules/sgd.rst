.. _sgd:

========================================
随机梯度下降
========================================

.. currentmodule:: sklearn.linear_model

**Stochastic Gradient Descent (SGD)** 是一种简单但又非常高效的方式判别式学习方法，比如凸损失函数的线性分类器如
`Support Vector Machines
<http://en.wikipedia.org/wiki/Support_vector_machine>`_ 和 `Logistic
Regression <http://en.wikipedia.org/wiki/Logistic_regression>`_.
虽然SGD已经在机器学习社区出现很长时间，但是在近期在大规模机器学习上受到了相当大数量的关注。

SGD 已经被成功应用到大规模和稀疏机器学习问题上，通常为文本分类和自然语言处理。如果给定数据是稀疏的，那么该模块中的分类器
很容易把问题规模缩放到超过10^5训练样本和超过10^5的特征数量。

SGD的优势如下：

    + 高效性.

    + 容易实现 (lots of opportunities for code tuning大量代码调整的机会).

SGD缺点如下：
    
    + SGD需要许多超参数,比如正则化参数、迭代次数

    + SGD 对特征规模比较敏感(应该是特征维数)

分类
==============

.. warning::

  请确保在拟合模型之前把训练数据打乱(shuffle)或者使用 ``shuffle=True`` 设置项来在每次迭代后打乱训练数据。

类 :class:`SGDClassifier` 实现了一个简单的随机梯度下降的程序，该程序支持分类中不同的损失函数和罚项

.. figure:: ../auto_examples/linear_model/images/plot_sgd_separating_hyperplane_001.png
   :target: ../auto_examples/linear_model/plot_sgd_separating_hyperplane.html
   :align: center
   :scale: 75

同其他分类器一样，SGD需要拟合两个数组(向量): X为存储训练样本的数组，大小为[n_samples,n_features]，另一个是Y,大小为[n_samples],
用来存放对于每个输入的目标值(或者类标label) ::

    >>> from sklearn.linear_model import SGDClassifier
    >>> X = [[0., 0.], [1., 1.]]
    >>> y = [0, 1]
    >>> clf = SGDClassifier(loss="hinge", penalty="l2")
    >>> clf.fit(X, y)
    SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
           eta0=0.0, fit_intercept=True, l1_ratio=0.15,
           learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1,
           penalty='l2', power_t=0.5, random_state=None, shuffle=True,
           verbose=0, warm_start=False)


拟合之后，模型就可以用来预测新的输入::

    >>> clf.predict([[2., 2.]])
    array([1])

SGD为训练数据拟合了一个线性模型。成员变量 ``coef_`` 存储的是模型的参数::

    >>> clf.coef_                                         # doctest: +ELLIPSIS
    array([[ 9.9...,  9.9...]])

成员变量 ``intercept_`` 存储的是截距 (又称为 offset 或者 bias,偏置)::

    >>> clf.intercept_                                    # doctest: +ELLIPSIS
    array([-9.9...])

无论模型是否使用截距，比如 一个有偏置的超平面，是由 ``fit_intercept`` 参数来控制(待校正)。

获取到超平面的符号距离使用 :meth:`SGDClassifier.decision_function`::

    >>> clf.decision_function([[2., 2.]])                 # doctest: +ELLIPSIS
    array([ 29.6...])

具体的损失函数可以通过 ``loss`` 参数来设置。:class:`SGDClassifier` 支持以下几种损失函数:

  * ``loss="hinge"``: (soft-margin) linear Support Vector Machine,
  * ``loss="modified_huber"``: smoothed hinge loss,
  * ``loss="log"``: logistic regression,
  * and all regression losses below.

上述中前两个损失函数lazy的，它们只有在某个样本违反了margin（间隔）限制才会更新模型参数，这样是的训练过程非常有效，并且可以应用在稀疏
模型上，甚至当使用了L2罚项的时候。

 使用 ``loss="log"`` 或者 ``loss="modified_huber"`` 启用
``predict_proba`` 方法,该方法给出了对于每个样本 :math:`x` 的概率估计 :math:`P(y|x)` 的一个向量::

    >>> clf = SGDClassifier(loss="log").fit(X, y)
    >>> clf.predict_proba([[1., 1.]])                      # doctest: +ELLIPSIS
    array([[ 0.00...,  0.99...]])

具体的罚项可以通过 ``penalty`` 参数。SGD支持一下几种罚项:

  * ``penalty="l2"``: L2 norm penalty on ``coef_``.
  * ``penalty="l1"``: L1 norm penalty on ``coef_``.
  * ``penalty="elasticnet"``: Convex combination of L2 and L1;
    ``(1 - l1_ratio) * L2 + l1_ratio * L1``.

默认的设置是 ``penalty="l2"``。L1罚项会导致稀疏的解，使大多数稀疏为0。弹性网络解决了当属性高度相关情况下L1罚项的不足。参数
 ``l1_ratio`` 控制 L1 和 L2 罚项的凸组合。

:class:`SGDClassifier` 通过组合多个“one versus all(OVA)”形式的二分类器来支持多类分类。
对于 :math:`K` 类中每个类别，二分类器通过判别该类和其它 :math:`K-1` 类来学习。在测试阶段，
我们计算为每个分类器计算其置信度得分（比如 与超平面的符号距离）。下图说明了OVA方式在iris数据集上的情况。
虚线表示三个OVA分类器;背景颜色显示了由三个分类器诱导的决策面。

.. figure:: ../auto_examples/linear_model/images/plot_sgd_iris_001.png
   :target: ../auto_examples/linear_model/plot_sgd_iris.html
   :align: center
   :scale: 75

在多分类问题中  ``coef_`` 是一个``shape=[n_classes, n_features]`` 的二维数组 ,
, ``intercept_`` 是一个  ``shape=[n_classes]`` 的一维数组。 ``coef_`` 的第i行
存储对第i类的OVA分类器的权重向量。类别通过增序索引（参考属性 ``classes_``）。
请注意，原则上由于 ``loss="log"`` 和 ``loss="modified_huber"`` 允许创建
概率模型，所以这两项对于OVA(one-vs-all)分类更加合适。

:class:`SGDClassifier` 支持加权类别和加权实例(或者说加权的样本)，通过 
``class_weight`` 和 ``sample_weight`` 两个拟合参数。请看下述几个例子，
参考文档 :meth:`SGDClassifier.fit` 获取更多信息。

.. topic:: Examples:

 - :ref:`example_linear_model_plot_sgd_separating_hyperplane.py`,
 - :ref:`example_linear_model_plot_sgd_iris.py`
 - :ref:`example_linear_model_plot_sgd_weighted_samples.py`
 - :ref:`example_linear_model_plot_sgd_comparison.py`
 - :ref:`example_svm_plot_separating_hyperplane_unbalanced.py` (See the `Note`)

:class:`SGDClassifier` 支持平均SGD(ASGD).Averaging可以通过设置  ```average=True``` 来启用。
ASGD 通过计算普通SGD算法中每次迭代后每个样本的系数的平均值来处理。当使用ASGD时，学习率可以大很多甚至为常量，
在一些数据集上训练时速度加快。。

对于带logistic损失的分类，提供了另外一种带平均策略的SGD变体，使用了随机平均梯度算法（SAG,
详细参考论文：Minimizing Finite Sums with the Stochastic Average Gradient）。
实现的程序为 :class:`LogisticRegression`.

回归
==========

 :class:`SGDRegressor` 类实现了一个简单的随机梯度下降的学习算法的程序，该程序支持不同的损失函数和罚项
 来拟合线性回归模型。 :class:`SGDRegressor` 对于非常大的训练样本(>10.000)的回归问题是非常合适的。
 对于其他问题我们推荐 :class:`Ridge`,:class:`Lasso`, 或者 :class:`ElasticNet` 。

具体损失函数可以通过设置  ``loss`` 参数。 :class:`SGDRegressor` 支持以下几种损失函数:

  * ``loss="squared_loss"``: Ordinary least squares,
  * ``loss="huber"``: Huber loss for robust regression,
  * ``loss="epsilon_insensitive"``: linear Support Vector Regression.

Huber 和 epsilon-insensitive 损失函数可以用于鲁棒回归。insensitive区域的宽度可以
通过参数 ``epsilon`` 指定，该参数由目标变量的规模来决定。

:class:`SGDRegressor` 和  :class:`SGDClassifier` 一样支持平均SGD。Averaging
可以通过设置 ```average=True``` 来启用。

对于带平方损失和l2罚项的回归，提供了另外一个带平均策略的SGD的变体，使用了随机平均梯度算法(SAG),
实现程序为  :class:`Ridge` 。


稀疏数据上的随机梯度下降
==============================

.. note:: 稀疏实现和稠密实现结果有轻微不同，因为截距部分的收敛的学习率的影响。


对于以下格式 `scipy.sparse <http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.html>`_ 任意给定矩阵的稀疏数据有内建的支持方法。
然而，为了最大化效率应该使用CSR矩阵格式，定义在 `scipy.sparse.csr_matrix <http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>`_.

.. topic:: Examples:

 - :ref:`example_text_document_classification_20newsgroups.py`

复杂度
==========

SGD主要的优势是它的高效性，和训练样本的数量线性相关。如果 X 是一个大小为(n ,p)的矩阵，则训练的代价为
 :math:`O(k n \bar p)` ，其中K是迭代的次数(epochs), :math:`\bar p` 是每个样本中非零属性(每个维度)的平均个数。

然而，最新理论研究结果显示，为了获得一些期望的最优的精度并不会随着训练样本集的大小增加而增加运行时间。

Tips on Practical Use
=====================
  * 随机梯度下降对于特征的尺度非常敏感，所以强烈推荐尺度化数据。比如，把每个输入向量X内的属性尺度化到区间[0,1]或者[-1,+1]
    上，或者把X标准化为均值为0，方差为1的数据。请注意，*相同的* 尺度也必须应用到测试向量上以保证得到有意义的结果。上述可以通过
    类 :class:`StandardScaler` 来处理 :: 

      from sklearn.preprocessing import StandardScaler
      scaler = StandardScaler()
      scaler.fit(X_train)  # Don't cheat - fit only on training data
      X_train = scaler.transform(X_train)
      X_test = scaler.transform(X_test)  # apply same transformation to test data

    如果你的特征向量的属性中有固定的尺度（比如词频或者指示特征）,则不必进行尺度化。

  * 在使用 :class:`GridSearchCV` 时最好的做法是找到一个合适的
    正则化项 :math:`\alpha` 通常取值范围为 ``10.0**-np.arange(1,7)`` 。

  * Empirically, we found that SGD converges after observing
    approx. 10^6 training samples. Thus, a reasonable first guess
    for the number of iterations is ``n_iter = np.ceil(10**6 / n)``,
    where ``n`` is the size of the training set.

  * If you apply SGD to features extracted using PCA we found that
    it is often wise to scale the feature values by some constant `c`
    such that the average L2 norm of the training data equals one.

  * We found that Averaged SGD works best with a larger number of features
    and a higher eta0

.. topic:: References:

 * `"Efficient BackProp" <yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf>`_
   Y. LeCun, L. Bottou, G. Orr, K. Müller - In Neural Networks: Tricks
   of the Trade 1998.

.. _sgd_mathematical_formulation:

Mathematical formulation
========================

Given a set of training examples :math:`(x_1, y_1), \ldots, (x_n, y_n)` where
:math:`x_i \in \mathbf{R}^n` and :math:`y_i \in \{-1,1\}`, our goal is to
learn a linear scoring function :math:`f(x) = w^T x + b` with model parameters
:math:`w \in \mathbf{R}^m` and intercept :math:`b \in \mathbf{R}`. In order
to make predictions, we simply look at the sign of :math:`f(x)`.
A common choice to find the model parameters is by minimizing the regularized
training error given by

.. math::

    E(w,b) = \frac{1}{n}\sum_{i=1}^{n} L(y_i, f(x_i)) + \alpha R(w)

where :math:`L` is a loss function that measures model (mis)fit and
:math:`R` is a regularization term (aka penalty) that penalizes model
complexity; :math:`\alpha > 0` is a non-negative hyperparameter.

Different choices for :math:`L` entail different classifiers such as

   - Hinge: (soft-margin) Support Vector Machines.
   - Log:   Logistic Regression.
   - Least-Squares: Ridge Regression.
   - Epsilon-Insensitive: (soft-margin) Support Vector Regression.

All of the above loss functions can be regarded as an upper bound on the
misclassification error (Zero-one loss) as shown in the Figure below.

.. figure:: ../auto_examples/linear_model/images/plot_sgd_loss_functions_001.png
    :target: ../auto_examples/linear_model/plot_sgd_loss_functions.html
    :align: center
    :scale: 75

Popular choices for the regularization term :math:`R` include:

   - L2 norm: :math:`R(w) := \frac{1}{2} \sum_{i=1}^{n} w_i^2`,
   - L1 norm: :math:`R(w) := \sum_{i=1}^{n} |w_i|`, which leads to sparse
     solutions.
   - Elastic Net: :math:`R(w) := \frac{\rho}{2} \sum_{i=1}^{n} w_i^2 + (1-\rho) \sum_{i=1}^{n} |w_i|`, a convex combination of L2 and L1, where :math:`\rho` is given by ``1 - l1_ratio``.

The Figure below shows the contours of the different regularization terms
in the parameter space when :math:`R(w) = 1`.

.. figure:: ../auto_examples/linear_model/images/plot_sgd_penalties_001.png
    :target: ../auto_examples/linear_model/plot_sgd_penalties.html
    :align: center
    :scale: 75

SGD
---

Stochastic gradient descent is an optimization method for unconstrained
optimization problems. In contrast to (batch) gradient descent, SGD
approximates the true gradient of :math:`E(w,b)` by considering a
single training example at a time.

The class :class:`SGDClassifier` implements a first-order SGD learning
routine.  The algorithm iterates over the training examples and for each
example updates the model parameters according to the update rule given by

.. math::

    w \leftarrow w - \eta (\alpha \frac{\partial R(w)}{\partial w}
    + \frac{\partial L(w^T x_i + b, y_i)}{\partial w})

where :math:`\eta` is the learning rate which controls the step-size in
the parameter space.  The intercept :math:`b` is updated similarly but
without regularization.

The learning rate :math:`\eta` can be either constant or gradually decaying. For
classification, the default learning rate schedule (``learning_rate='optimal'``)
is given by

.. math::

    \eta^{(t)} = \frac {1}{\alpha  (t_0 + t)}

where :math:`t` is the time step (there are a total of `n_samples * n_iter`
time steps), :math:`t_0` is determined based on a heuristic proposed by Léon Bottou
such that the expected initial updates are comparable with the expected
size of the weights (this assuming that the norm of the training samples is
approx. 1). The exact definition can be found in ``_init_t`` in :class:`BaseSGD`.


For regression the default learning rate schedule is inverse scaling
(``learning_rate='invscaling'``), given by

.. math::

    \eta^{(t)} = \frac{eta_0}{t^{power\_t}}

where :math:`eta_0` and :math:`power\_t` are hyperparameters chosen by the
user via ``eta0`` and ``power_t``, resp.

For a constant learning rate use ``learning_rate='constant'`` and use ``eta0``
to specify the learning rate.

The model parameters can be accessed through the members ``coef_`` and
``intercept_``:

     - Member ``coef_`` holds the weights :math:`w`

     - Member ``intercept_`` holds :math:`b`

.. topic:: References:

 * `"Solving large scale linear prediction problems using stochastic
   gradient descent algorithms"
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.58.7377>`_
   T. Zhang - In Proceedings of ICML '04.

 * `"Regularization and variable selection via the elastic net"
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.124.4696>`_
   H. Zou, T. Hastie - Journal of the Royal Statistical Society Series B,
   67 (2), 301-320.

 * `"Towards Optimal One Pass Large Scale Learning with
   Averaged Stochastic Gradient Descent"
   <http://arxiv.org/pdf/1107.2490v2.pdf>`_
   Xu, Wei


Implementation details
======================

The implementation of SGD is influenced by the `Stochastic Gradient SVM
<http://leon.bottou.org/projects/sgd>`_  of Léon Bottou. Similar to SvmSGD,
the weight vector is represented as the product of a scalar and a vector
which allows an efficient weight update in the case of L2 regularization.
In the case of sparse feature vectors, the intercept is updated with a
smaller learning rate (multiplied by 0.01) to account for the fact that
it is updated more frequently. Training examples are picked up sequentially
and the learning rate is lowered after each observed example. We adopted the
learning rate schedule from Shalev-Shwartz et al. 2007.
For multi-class classification, a "one versus all" approach is used.
We use the truncated gradient algorithm proposed by Tsuruoka et al. 2009
for L1 regularization (and the Elastic Net).
The code is written in Cython.

.. topic:: References:

 * `"Stochastic Gradient Descent" <http://leon.bottou.org/projects/sgd>`_ L. Bottou - Website, 2010.

 * `"The Tradeoffs of Large Scale Machine Learning" <http://leon.bottou.org/slides/largescale/lstut.pdf>`_ L. Bottou - Website, 2011.

 * `"Pegasos: Primal estimated sub-gradient solver for svm"
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.74.8513>`_
   S. Shalev-Shwartz, Y. Singer, N. Srebro - In Proceedings of ICML '07.

 * `"Stochastic gradient descent training for l1-regularized log-linear models with cumulative penalty"
   <http://www.aclweb.org/anthology/P/P09/P09-1054.pdf>`_
   Y. Tsuruoka, J. Tsujii, S. Ananiadou -  In Proceedings of the AFNLP/ACL '09.
