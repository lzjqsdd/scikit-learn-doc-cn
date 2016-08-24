# scikit-learn-doc-cn

[![Build Status](https://travis-ci.org/lzjqsdd/scikit-learn-doc-cn.svg?branch=master)](https://travis-ci.org/lzjqsdd/scikit-learn-doc-cn) ![Build Status](https://img.shields.io/badge/sklearn-0.17stable-brightgreen.svg)	![Build Status](https://img.shields.io/badge/python-2.7.6-brightgreen.svg)

sklearn库作为目前机器学习非常流行的python库，一个易读的文档更有助于工具库的理解和使用，虽然做机器学习方面的学生和工程师阅读英文并没有很大压力，但是在阅读速度上还是会有些影响。
寻找已久没找到相关的中文文档，而且翻译的过程也是对知识熟悉的过程，您可以在学习某一个章节的过程顺便翻译一下就可以贡献自己的力量。

*欢迎大家踊跃加入！如果有更好的翻译组织形式也欢迎提出！*

*中文文档地址:[sklearn.lzjqsdd.com](http://sklearn.lzjqsdd.com)

# 翻译计划

| 图标  |  状态 |
|:----------------|:------------:|
| ![Build Status](https://img.shields.io/badge/translate-doing-brightgreen.svg)| 翻译中|
| ![Build Status](https://img.shields.io/badge/translate-done-blue.svg)| 翻译结束|
| ![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)| 暂未开始 |


## 第一阶段
相关算法示例程序的翻译，位于modules下，具体列表如下：


| 文件                | 翻译状态        | 贡献者
|:------------------- |:-------------| :-------:|
| linear_model.rst    | ![Build Status](https://img.shields.io/badge/translate-done-blue.svg) | lzjqsdd |
| biclustering.rst    | ![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)    | 		|
| calibration.rst | ![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg) ||
| classes.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg) ||
| clustering.rst | ![Build Status](https://img.shields.io/badge/translate-doing-brightgreen.svg)|huangbinapple|
| computational_performance.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| covariance.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| cross_decomposition.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| cross_validation.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| decomposition.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| density.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| dp-derivation.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| ensemble.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| feature_extraction.rst |![Build Status](https://img.shields.io/badge/translate-doing-brightgreen.svg)| minoriwww |
| feature_selection.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| gaussian_process.rst |![Build Status](https://img.shields.io/badge/translate-doing-brightgreen.svg)|alingse|
| grid_search.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| isotonic.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| kernel_approximation.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| kernel_ridge.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| label_propagation.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| lda_qda.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| learning_curve.rst |![Build Status]((https://img.shields.io/badge/translate-doing-brightgreen.svg))| taoyizhi68 |
| manifold.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| metrics.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| mixture.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| model_evaluation.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| model_persistence.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| multiclass.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| naive_bayes.rst |![Build Status](https://img.shields.io/badge/translate-done-blue.svg)| minoriwww |
| neighbors.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| neural_networks_supervised.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| neural_networks_unsupervised.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| outlier_detection.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| pipeline.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| preprocessing.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| preprocessing_targets.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| random_projection.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| scaling_strategies.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| sgd.rst |![Build Status](https://img.shields.io/badge/translate-doing-brightgreen.svg)|lzjqsdd|
| tree.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)||
| unsupervised_reduction.rst |![Build Status](https://img.shields.io/badge/translate-undo-lightgray.svg)| |

### 任务申领
首先clone该项目，修改README中要申领的翻译内容的状态，提交pull request,接受之后即可开始翻译工作。
翻译结束后确保可正常编译成html，然后只提交rst文件的修改，不要添加sphinx生成的文件。
**所有翻译后的文档以同名的方式添加到translate/同目录文件夹下,例如：**  

    svm.rst的翻译文档 提交到项目translate/modules/svm.rst下,翻译完成之后覆盖doc/modules/svm.rst。



## 阶段二
官方框架翻译

#配置及编译

安装必要的环境:

	sudo pip install numpy
    sudo pip install scipy
    sudo pip install sphinx
    #上述为依赖的包
    sudo pip install -U scikit-learn

生成html（和官网web页一样）

    make html

生成文件会在在_build/html目录下:

如果要生成PDF手册的话：

    make latexpdf

部署gh-pages：

	由于Sphinx生成的html有自己的静态资源，需要在gh-pages分支加入.nojekyll文件。

配置中其他的问题：
	[Issue](https://github.com/lzjqsdd/scikit-learn-doc-cn/blob/master/issue.md)
