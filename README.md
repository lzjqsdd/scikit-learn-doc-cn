# scikit-learn-doc-cn

sklearn库作为目前机器学习非常流行的python库，一个易读的文档更有助于工具库的理解和使用，虽然做机器学习方面的学生和工程师阅读英文并没有很大压力，但是在阅读速度上还是会有些影响。
寻找已久没找到相关的中文文档，而且翻译的过程也是对知识熟悉的过程，您可以在学习某一个章节的过程顺便翻译一下就可以贡献自己的力量。

*欢迎大家踊跃加入！*

# 翻译计划
## 第一阶段
相关算法示例程序的翻译，位于modules下，具体列表如下：
![Build Status](https://img.shields.io/badge/translate-working-brightgreen.svg) 翻译中

![Build Status](https://img.shields.io/badge/translate-done-blue.svg) 翻译结束

![Build Status](https://img.shields.io/badge/translate-notstarted-lightgray.svg) 暂未开始


- [ ] linear_model.rst ![Build Status](https://img.shields.io/badge/translate-working-brightgreen.svg)
- [ ] biclustering.rst
- [ ] calibration.rst
- [ ] classes.rst
- [ ] clustering.rst
- [ ] computational_performance.rst
- [ ] covariance.rst
- [ ] cross_decomposition.rst
- [ ] cross_validation.rst
- [ ] decomposition.rst
- [ ] density.rst
- [ ] dp-derivation.rst
- [ ] ensemble.rst
- [ ] feature_extraction.rst
- [ ] feature_selection.rst
- [ ] gaussian_process.rst
- [ ] grid_search.rst
- [ ] isotonic.rst
- [ ] kernel_approximation.rst
- [ ] kernel_ridge.rst
- [ ] label_propagation.rst
- [ ] lda_qda.rst
- [ ] learning_curve.rst
- [ ] manifold.rst
- [ ] metrics.rst
- [ ] mixture.rst
- [ ] model_evaluation.rst
- [ ] model_persistence.rst
- [ ] multiclass.rst
- [ ] naive_bayes.rst
- [ ] neighbors.rst
- [ ] neural_networks_supervised.rst
- [ ] neural_networks_unsupervised.rst
- [ ] outlier_detection.rst
- [ ] pipeline.rst
- [ ] preprocessing.rst
- [ ] preprocessing_targets.rst
- [ ] random_projection.rst
- [ ] scaling_strategies.rst
- [ ] sgd.rst
- [ ] svm.rst
- [ ] tree.rst
- [ ] unsupervised_reduction.rst

**所有翻译后的文档以同名的方式添加到translate/同目录文件夹下,例如：**  

    svm.rst的翻译文档 提交到项目translate/modules/svm.rst下,翻译完成之后覆盖doc/modules/svm.rst。


## 阶段二
官方框架翻译

#编译

生成html（和官网web页一样）

    make html

生成文件会在在_build/html目录下:

如果要生成PDF手册的话：

    make latexpdf

