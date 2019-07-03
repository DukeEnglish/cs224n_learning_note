# 第二章 笔记

## 2.1 note：

主要内容：

1. Global Vectors for Word Representation ( GloVe)
2. 内外部评估
3. analogy evalution 任务的超参影响
4. 人类评估和word vector距离
5. 使用context来处理词汇的模糊问题
6. window classification

### 2.1.1 GloVe

由于之前提到的模型一个是基于计数和矩阵因式分解的，它们有效地利用了全局统计信息，但是主要时候捕捉了词相似度，但是在词语推理上表现一般，这说明它达到了局部最优向量空间结构。
另一个是基于浅层窗口的（skip-gram 以及 CBOW）方法，这些方法尽管不住红了语法规则，但是却不能利用全局共现统计信息。
GloVe利用了权重最小二乘法，利用到了全局的信息，所以就有效的利用到了统计信息。其产生了意义丰富的向量空间子结构。【简单来说，skig-gram混合CBOW都是基于一个窗口去预测，但是GloVe是基于给定全文去 预测的。】

GloVe consists of weighted least squares model that trains on global word-word co-occurrence counts and thus makes efficient use of statistics.



## 2.2 video and slides

本次lecture讲的内容不多，主要是回顾了gensim的包，讲解了一下词向量，以及评估方法等。这些内容会在assignment1 中有所体现，可以通过完成a1巩固学习。

概念：

- GD：所有的数据过一遍进行更新
- SGD：sample一个window进行更新（一条数据更新一次，在我们的问题里面就是一个window一次）
- mini-batch：一个batch的数据的进行更新



### 2.3 其他



这里出现的几个paper都比较经典，不过其实都被在blog或者各种地方解读过了，是否需要读，完全取决于自己。



















