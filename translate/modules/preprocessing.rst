.. _preprocessing:

==================
数据预处理
==================

.. currentmodule:: sklearn.preprocessing

``sklearn.preprocessing``包为用户提供了多个工具函数和类，用于将原始特征转换成更适于项目后期学习的特征表示。

.. _preprocessing_scaler:

标准化、去均值、方差缩放(variance scaling)
=====================================================

数据集的** 标准化 **对于在scikit中的大部分机器学习算法来说都是一种** 常规要求 ** 。如果单个特征没有或多或少地接近于标准正态分布：** 零均值和单位方差 **的高斯分布，那么它可能并不能在项目中表现出很好的性能。

在实际情况中,我们经常忽略特征的分布形状，直接经过去均值来对某个特征进行中心化，再通过除以非常量特征(non-constant features)的标准差进行缩放。

例如, 许多学习算法中目标函数的基础都是假设所有的特征都是零均值并且具有同一阶数上的方差(比如径向基函数、支持向量机以及L1\L2正则化项等)。如果某个特征的方差比其他特征大几个数量级，那么它就会在学习算法中占据主导位置，导致学习器并不能像我们说期望的那样，从其他特征中学习。


工具函数:func:`scale` 为数组形状的数据集的标准化提供了一个快捷实现::

  >>> from sklearn import preprocessing
  >>> import numpy as np
  >>> X = np.array([[ 1., -1.,  2.],
  ...               [ 2.,  0.,  0.],
  ...               [ 0.,  1., -1.]])
  >>> X_scaled = preprocessing.scale(X)

  >>> X_scaled                                          # doctest: +ELLIPSIS
  array([[ 0.  ..., -1.22...,  1.33...],
         [ 1.22...,  0.  ..., -0.26...],
         [-1.22...,  1.22..., -1.06...]])

..
        >>> import numpy as np
        >>> print_options = np.get_printoptions()
        >>> np.set_printoptions(suppress=True)

经过缩放的数据集具有零均值和标准方差::

  >>> X_scaled.mean(axis=0)
  array([ 0.,  0.,  0.])

  >>> X_scaled.std(axis=0)
  array([ 1.,  1.,  1.])

..    >>> print_options = np.set_printoptions(print_options)

``preprocessing``模块也提供了一个实用类:class:`StandardScaler` ,它使用 ``Transformer`` 接口在训练集上计算均值和标准差，以便于在后续的测试集上进行相同的缩放.
This class is hence suitable for use in the early steps of a :class:`sklearn.pipeline.Pipeline`::

  >>> scaler = preprocessing.StandardScaler().fit(X)
  >>> scaler
  StandardScaler(copy=True, with_mean=True, with_std=True)

  >>> scaler.mean_                                      # doctest: +ELLIPSIS
  array([ 1. ...,  0. ...,  0.33...])

  >>> scaler.scale_                                       # doctest: +ELLIPSIS
  array([ 0.81...,  0.81...,  1.24...])

  >>> scaler.transform(X)                               # doctest: +ELLIPSIS
  array([[ 0.  ..., -1.22...,  1.33...],
         [ 1.22...,  0.  ..., -0.26...],
         [-1.22...,  1.22..., -1.06...]])

缩放类对象可以在新的数据上实现和训练集相同缩放操作::

  >>> scaler.transform([[-1.,  1., 0.]])                # doctest: +ELLIPSIS
  array([[-2.44...,  1.22..., -0.26...]])

你也可以通过在构造函数:class:`StandardScaler`中传入参数``with_mean=False`` 或者``with_std=False``来取消中心化或缩放操作。


特征缩放至特定范围
---------------------------

另外一个可选的缩放操作是将特征缩放至给定的最小、最大值范围，经常是[0,1]。
或者也可以将每个特征的最大绝对值转换至单位大小。这两类操作可以分别通过使用:class:`MinMaxScaler`或者:class:`MaxAbsScaler`实现。

The motivation to use this scaling include robustness to very small
standard deviations of features and preserving zero entries in sparse data.

下面这个例子展示了将一个简单的数据矩阵缩放至``[0, 1]``范围::

  >>> X_train = np.array([[ 1., -1.,  2.],
  ...                     [ 2.,  0.,  0.],
  ...                     [ 0.,  1., -1.]])
  ...
  >>> min_max_scaler = preprocessing.MinMaxScaler()
  >>> X_train_minmax = min_max_scaler.fit_transform(X_train)
  >>> X_train_minmax
  array([[ 0.5       ,  0.        ,  1.        ],
         [ 1.        ,  0.5       ,  0.33333333],
         [ 0.        ,  1.        ,  0.        ]])

同样的转换实例可以被用与在训练过程中不可见的测试数据:实现和训练数据一致的缩放和移位操作::

  >>> X_test = np.array([[ -3., -1.,  4.]])
  >>> X_test_minmax = min_max_scaler.transform(X_test)
  >>> X_test_minmax
  array([[-1.5       ,  0.        ,  1.66666667]])

你也可以通过查看缩放器(scaler)的属性，来观察在训练集中学习到的转换操作的基本性质::

  >>> min_max_scaler.scale_                             # doctest: +ELLIPSIS
  array([ 0.5       ,  0.5       ,  0.33...])

  >>> min_max_scaler.min_                               # doctest: +ELLIPSIS
  array([ 0.        ,  0.5       ,  0.33...])

如果:class:`MinMaxScaler`被提供了一个精确的``feature_range=(min, max)``，完整的公式是::

    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    X_scaled = X_std / (max - min) + min

:class:`MaxAbsScaler` 工作原理非常相似,但是它只通过除以每个特征的最大值将训练数据特征缩放至 ``[-1, 1]``。这就意味着，训练数据应该是已经零中心化或者是稀疏数据。
例子::用先前例子的数据实现最大绝对值缩放操作::

  >>> X_train = np.array([[ 1., -1.,  2.],
  ...                     [ 2.,  0.,  0.],
  ...                     [ 0.,  1., -1.]])
  ...
  >>> max_abs_scaler = preprocessing.MaxAbsScaler()
  >>> X_train_maxabs = max_abs_scaler.fit_transform(X_train)
  >>> X_train_maxabs                # doctest +NORMALIZE_WHITESPACE^
  array([[ 0.5, -1. ,  1. ],
         [ 1. ,  0. ,  0. ],
         [ 0. ,  1. , -0.5]])
  >>> X_test = np.array([[ -3., -1.,  4.]])
  >>> X_test_maxabs = max_abs_scaler.transform(X_test)
  >>> X_test_maxabs                 # doctest: +NORMALIZE_WHITESPACE
  array([[-1.5, -1. ,  2. ]])
  >>> max_abs_scaler.scale_         # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
  array([ 2.,  1.,  2.])


如果你并不想创造一个对象，那么你可以通过使用:func:`scale`, :func:`minmax_scale`, :func:`maxabs_scale`来快速实现缩放操作。


稀疏数据缩放
-------------------
稀疏数据的中心化会破坏数据中的稀疏结构，因此很少有一个比较明智的实现方式。但是，对稀疏输入进行缩放操作是有意义的，特别是当特征的数值并不在同一个量级上面的时候。
:class:`MaxAbsScaler` 和:func:`maxabs_scale` 是稀疏数据缩放比较推荐的实现方式。
但是，:func:`scale` 和 :class:`StandardScaler` 可以接收``scipy.sparse``矩阵作为输入,只要将参数``with_centering=False``传入构造函数。否则程序将会出现异常 ``ValueError`` 。
因为默认的中心化操作会破坏数据的稀疏性，并导致大量的内存占用。
:class:`RobustScaler` 并不适用于稀疏数据，但是你可以在稀疏输入上使用the ``transform``方法。

注意，缩放器既可以接收行压缩稀疏数据也可以是列压缩稀疏数据(参见``scipy.sparse.csr_matrix``和``scipy.sparse.csc_matrix``)。 其他形式的稀疏输入**会被转换为行压缩稀疏表示**。为了避免不必要的内存复制，推荐在顶层使用CSR或CSC表示。

最后，如果已经中心化的数据并不是很大，可以选择使用''toarray''方法将稀疏矩阵转换为数组。


含异常值数据缩放
--------------------------

如果你的数据包含较多的异常值，使用均值和方差缩放可能并不是一个很好的选择。在这种情况下，你可以使用
:func:`robust_scale` 和:class:`RobustScaler` 作为替代。它们使用更加鲁棒的中心和范围估计来缩放你的数据。


.. topic:: 参考:

  Further discussion on the importance of centering and scaling data is
  available on this FAQ: `Should I normalize/standardize/rescale the data?
  <http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html>`_

.. topic:: 缩放 vs 白化

  有时候单独地中心化和缩放并不是足够的，因为一些下层的学习算法还可能假设特征之间是线性独立的。

  为了解决这个问题，你可以使用:class:`sklearn.decomposition.PCA`
  或者:class:`sklearn.decomposition.RandomizedPCA` 带参数``whiten=True``来进一步移除特征之间的线性相关性。

.. topic:: 回归中的目标变量缩放

    :func:`scale` and :class:`StandardScaler`可以对一维数组进行使用。这对于在回归当中的目标值或响应变量进行缩放时非常有效的。

.. _kernel_centering:

核矩阵中心化
-------------------------

如果你有一个使用:math:`phi`定义的核矩阵 :math:`K`，用以计算特征空间上的内积，
:class:`KernelCenterer`可以用于核矩阵变换，变换后在特征空间的内积有一个已经去均值的:math:`phi`来定义。

.. _preprocessing_normalization:

规范化
=============

**规范化是使单个样本具有单位范数的缩放操作。** 这个在你使用二次型，比如点积或者其他核去定量计算任意样本对之间的相似性时是非常有用的。

规范化是向量空间模型
<http://en.wikipedia.org/wiki/Vector_Space_Model>`_的基本假设，经常在文本分类和聚类当中使用。

方法 :func:`normalize` 为在数组类型的数据集提供了规范化的快速实现,可以选择``l1``或``l2``
范数::

  >>> X = [[ 1., -1.,  2.],
  ...      [ 2.,  0.,  0.],
  ...      [ 0.,  1., -1.]]
  >>> X_normalized = preprocessing.normalize(X, norm='l2')

  >>> X_normalized                                      # doctest: +ELLIPSIS
  array([[ 0.40..., -0.40...,  0.81...],
         [ 1.  ...,  0.  ...,  0.  ...],
         [ 0.  ...,  0.70..., -0.70...]])

``preprocessing``模块也提供了实用类
:class:`Normalizer` ，通过使用接口
``Transformer`` 来实现相同的操作。(在这里``fit``方法并没有作用:
因为规范化类在面对不同的样本数据时是无状态独立的)。

This class is hence suitable for use in the early steps of a
:class:`sklearn.pipeline.Pipeline`::

  >>> normalizer = preprocessing.Normalizer().fit(X)  # fit函数没有任何效果
  >>> normalizer
  Normalizer(copy=True, norm='l2')


规范化实例可以在任意样本向量上实现::

  >>> normalizer.transform(X)                            # doctest: +ELLIPSIS
  array([[ 0.40..., -0.40...,  0.81...],
         [ 1.  ...,  0.  ...,  0.  ...],
         [ 0.  ...,  0.70..., -0.70...]])

  >>> normalizer.transform([[-1.,  1., 0.]])             # doctest: +ELLIPSIS
  array([[-0.70...,  0.70...,  0.  ...]])


.. topic:: 稀疏输入

  :func:`normalize` and :class:`Normalizer`**既可以接收稠密数组也可以接收scipy.sparse的稀疏矩阵作为输入。**

  输入的稀疏数据**会被转化为行压缩稀疏表示** (参见``scipy.sparse.csr_matrix``)，然后再被送进Cython程序。为了避免不必要的内存复制，推荐在顶层使用CSR或CSC表示。

.. _preprocessing_binarization:

二值化
============

特征二值化
--------------------

**特征二值化是将数值型特征变成布尔型特征**。这对于当下层概率估计器假设输入数据是多变量` 伯努利分布<http://en.wikipedia.org/wiki/Bernoulli_distribution>`_ 时是非常有效的。
例如， :class:`sklearn.neural_network.BernoulliRBM`。

在文本处理中，即使规范化计数(a.k.a. term frequencies)或者TF-IDF数值特征在实际情况中性能略胜一筹，使用布尔型特征仍旧是是非常常见的(很大可能可以简化概率推演)。

和:class:`Normalizer`一样，
:class:`Binarizer` is meant to be used in the early stages of
:class:`sklearn.pipeline.Pipeline`. ``fit`` 也并没有作用，
因为样本之间都是独立对待的::

  >>> X = [[ 1., -1.,  2.],
  ...      [ 2.,  0.,  0.],
  ...      [ 0.,  1., -1.]]

  >>> binarizer = preprocessing.Binarizer().fit(X)  # fit does nothing
  >>> binarizer
  Binarizer(copy=True, threshold=0.0)

  >>> binarizer.transform(X)
  array([[ 1.,  0.,  1.],
         [ 1.,  0.,  0.],
         [ 0.,  1.,  0.]])

可以改变二值器的阈值::

  >>> binarizer = preprocessing.Binarizer(threshold=1.1)
  >>> binarizer.transform(X)
  array([[ 0.,  0.,  1.],
         [ 1.,  0.,  0.],
         [ 0.,  0.,  0.]])

跟:class:`StandardScaler`和:class:`Normalizer`类一样，模块也提供了二值化方法:func:`binarize`，以便不需要转换接口时使用。

.. topic::稀疏输入

  :func:`binarize`和:class:`Binarizer`**既可以接收稠密数组也可以接收scipy.sparse的稀疏矩阵作为输入。**

  输入的稀疏数据**会被转化为行压缩稀疏表示** (参见``scipy.sparse.csr_matrix``)，然后再被送进Cython程序。为了避免不必要的内存复制，推荐在顶层使用CSR或CSC表示。


.. _preprocessing_categorical_features:

分类特征编码
=============================
特征更多的时候是分类特征，而不是连续的数值特征。
比如一个人的特征可以是``["male", "female"]``，
``["from Europe", "from US", "from Asia"]``，
``["uses Firefox", "uses Chrome", "uses Safari", "uses Internet Explorer"]``。
这样的特征可以高效的编码成整数，例如
``["male", "from US", "uses Internet Explorer"]``可以表示成
``[0, 1, 3]``，``["female", "from Asia", "uses Chrome"]``就是``[1, 2, 1]``。

这个的整数特征表示并不能在scikit-learn的估计器中直接使用，因为这样的连续输入，估计器会认为类别之间是有序的，但实际却是无序的。(例如：浏览器的类别数据则是任意排序的)。

一个将分类特征转换成scikit-learn估计器可用特征的可选方法是使用one-of-K或者one-hot编码,该方法是:class:`OneHotEncoder`的一个实现。该方法将每个类别特征的 ``m`` 可能值转换成``m``个二进制特征值，当然只有一个是激活值。

续上例::

  >>> enc = preprocessing.OneHotEncoder()
  >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])  # doctest: +ELLIPSIS
  OneHotEncoder(categorical_features='all', dtype=<... 'float'>,
         handle_unknown='error', n_values='auto', sparse=True)
  >>> enc.transform([[0, 1, 3]]).toarray()
  array([[ 1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.]])

默认情况下，每个特征使用几维的数值由数据集自动推断。当然，你也可以通过使用参数``n_values``来精确指定。
在我们的例子数据集中，有两个可能得性别类别，三个洲，四个网络浏览器。接着，我们训练编码算法，并用来对一个样本数据进行转换。
在结果中，前两个数值是性别编码，紧接着的三个数值是洲编码，最后的四个数值是浏览器编码。

查询:参考:`dict_feature_extraction` 用于将类别数据标示为字典型数据，而不是整数。

.. _imputation:

缺失值处理（Imputation）
============================

因为各种各样的原因，真实世界中的许多数据集都包含缺失数据，这类数据经常被编码成空格、NaNs，或者是其他的占位符。但是这样的数据集并不能scikit-learn学习算法兼容，因为大多的学习算法都默认假设数组中的元素都是数值，因而所有的元素都有自己的意义。
使用不完整的数据集的一个基本策略就是舍弃掉整行或整列包含缺失值的数据。但是这样就付出了舍弃可能有价值数据（即使是不完整的 ）的代价。
处理缺失数值的一个更好的策略就是从已有的数据推断出缺失的数值。

:class:`Imputer`类提供了缺失数值处理的基本策略，比如使用缺失数值所在行或列的均值、中位数、众数来替代缺失值。该类也兼容不同的缺失值编码。

接下来是一个如何替换缺失值的简单示例，缺失值被编码为``np.nan``, 使用包含缺失值的列的均值来替换缺失值。

    >>> import numpy as np
    >>> from sklearn.preprocessing import Imputer
    >>> imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    >>> imp.fit([[1, 2], [np.nan, 3], [7, 6]])
    Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)
    >>> X = [[np.nan, 2], [6, np.nan], [7, 6]]
    >>> print(imp.transform(X))                           # doctest: +ELLIPSIS
    [[ 4.          2.        ]
     [ 6.          3.666...]
     [ 7.          6.        ]]

:class:`Imputer` 类也支持稀疏矩阵::

    >>> import scipy.sparse as sp
    >>> X = sp.csc_matrix([[1, 2], [0, 3], [7, 6]])
    >>> imp = Imputer(missing_values=0, strategy='mean', axis=0)
    >>> imp.fit(X)
    Imputer(axis=0, copy=True, missing_values=0, strategy='mean', verbose=0)
    >>> X_test = sp.csc_matrix([[0, 2], [6, 0], [7, 6]])
    >>> print(imp.transform(X_test))                      # doctest: +ELLIPSIS
    [[ 4.          2.        ]
     [ 6.          3.666...]
     [ 7.          6.        ]]

注意，在这里，缺失数据被编码为0， and are thus implicitly stored
in the matrix. 这种方式用在当缺失数据比观察数据更多的情况时是非常合适的。

:class:`Imputer` can be used in a Pipeline as a way to build a composite
estimator that supports imputation. See :ref:`example_missing_values.py`

.. _polynomial_features:

多项式特征生成
==============================

很多情况下，考虑输入数据中的非线性特征来增加模型的复杂性是非常有效的。一个简单常用的方法就是使用多项式特征，它能捕捉到特征中高阶和相互作用的项。
:class:`PolynomialFeatures`类中可以实现该功能::

    >>> import numpy as np
    >>> from sklearn.preprocessing import PolynomialFeatures
    >>> X = np.arange(6).reshape(3, 2)
    >>> X                                                 # doctest: +ELLIPSIS
    array([[0, 1],
           [2, 3],
           [4, 5]])
    >>> poly = PolynomialFeatures(2)
    >>> poly.fit_transform(X)                             # doctest: +ELLIPSIS
    array([[  1.,   0.,   1.,   0.,   0.,   1.],
           [  1.,   2.,   3.,   4.,   6.,   9.],
           [  1.,   4.,   5.,  16.,  20.,  25.]])

特征向量X从:math:`(X_1, X_2)` 被转换成:math:`(1, X_1, X_2, X_1^2, X_1X_2, X_2^2)`。

在一些情况中，我们只需要特征中的相互作用项(interaction terms)，它可以通过传入参数``interaction_only=True``获得::

    >>> X = np.arange(9).reshape(3, 3)
    >>> X                                                 # doctest: +ELLIPSIS
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> poly = PolynomialFeatures(degree=3, interaction_only=True)
    >>> poly.fit_transform(X)                             # doctest: +ELLIPSIS
    array([[   1.,    0.,    1.,    2.,    0.,    0.,    2.,    0.],
           [   1.,    3.,    4.,    5.,   12.,   15.,   20.,   60.],
           [   1.,    6.,    7.,    8.,   42.,   48.,   56.,  336.]])

特征向量X从:math:`(X_1, X_2, X_3)` 被转换成:math:`(1, X_1, X_2, X_3, X_1X_2, X_1X_3, X_2X_3, X_1X_2X_3)`。

注意多项式特征被隐含地使用在`核方法<http://en.wikipedia.org/wiki/Kernel_method>`_ (例如， :class:`sklearn.svm.SVC`, :class:`sklearn.decomposition.KernelPCA`) :ref:`svm_kernels`.

See :ref:`example_linear_model_plot_polynomial_interpolation.py` for Ridge regression using created polynomial features.

装换器定制
===================

你可能经常需要将一个已经存在的python函数转换成转换器以便于在数据清理和预处理当中使用。你可以通过:class:`FunctionTransformer`类将任意一个函数转换成转换器。
例如，创建一个实现对数变换的转换器，可以这么做::

    >>> import numpy as np
    >>> from sklearn.preprocessing import FunctionTransformer
    >>> transformer = FunctionTransformer(np.log1p)
    >>> X = np.array([[0, 1], [2, 3]])
    >>> transformer.transform(X)
    array([[ 0.        ,  0.69314718],
           [ 1.09861229,  1.38629436]])

对于使用:class:`FunctionTransformer`去做特殊特征选择的完整代码示例可以:参考:`example_preprocessing_plot_function_transformer.py`
