觉得还是将自己的学习笔记做一个简单的记录比较好，不然回头又忘记了

# 第一课 word2vec

尽管现在bert啥的都火了起来，但是起源都在这里，万丈高楼平地起。需要回顾一下啦。很久没有碰DL了。

## 视频1

第一个视频没有什么难度，也没有什么concrete的内容，但是有很多老师自己的想法，心得，对于刚刚开始NLP，或者回头想要重新思考NLP本质的人来说还是有意义的。

没什么难度，主要是一个预热，那个作业不错，应该好好做一下。还有相关的reading amterial

#### 注意点

1. softmax：大化大值，小值>0。 hardmax：保留大值为1，其他都是0
2. Page27 why each word has two vector? Caz each word can be context word and center
3. 难点：context和center word的推导
4. 重点：材料的阅读，并且做好一定的review，完成assign1. 材料很细致，学过的再看一次，会有新理解。

## 材料

第一课的材料如下：

1. 除了视频和slides之外，有课程notes。如果跟不上，可以先看note，再听课。
2. 关于gensim，如果没有用过，可以看一下代码示例，这没什么难度，可以跳过，等之后用到的话再来看（我用过了，所以我跳过了）
3. 建议阅读：
   1. w2v的材料。这个是stanford毕业的学生写的，不得不说，解释的很到位，对新手很友好
   2. original word2vec paper
   3. negative sampling paper
4. 作业。这个也不难，可以自己做。我写的参考答案在a1里面。

在仔细地重新过自己学过的知识，根据每节课的内容。

### note

1. 主要内容：begins by intrduing the **concept of NLP** and the **problems NLP faces** today. Then discuss the concept of **representing words** as numeric vectors. Lastly discuss popular approaches to **designing word vectors**.
2. 摘抄要点：
   1. the word "rocket" refers to the concept of a rocket, and by extension can designate an instance of a rocket. 单词火箭可以指火箭的概念，但是也可以指火箭这个实例（东西）
3. 同时也给出了传统的SVD灯方法进行word表示
4. 第四章是重点

#### 第四章

我们创建一个model能一次学习一轮，且能在给定context的情况下编码一个word

idea就是设计一个参数就是word vector的模型

我们的任务就是给定一个object function（a.k.a cost/loss function）

​	**Word2vec** ：一个软件包，包括：

 - 两个algorithms：
   	- continous bags of words(CBOW): give context
   	- Skip-gram: give center word
 - 2 training methods:
   	- negative sampling: define an objective by sampling nagetive examples
   	- hierarchical softmax: defines an objective using an efficient tree structure to compute probabilities for all the vocabulary



##### 4.1 Language models (unigrams, bigrams, etc)

​	We think about a sequence of tokens having a probability, some models that could learn these probabilities "which will be introduced in the following sessions [CBOW, SGM, NS, HS]."

##### 4.2 Continuous Bag of words model (CBOW)

1. Set up our known parameters: sentence represented by one-hot vector
2. Input: one-hot vector or context  $$x^{c}$$ 
3. Output: $$y^{c}$$ . Because in CBOW model, we only have one output so  $$y^{c}$$ is the one-hot vector of the known center word [which means the label here is center word]
4. define the unkowns in our model

在这一章节中，详细地给出了CBOW模型的工作步骤。

最后我们得到了预测结果。然后使用cross-entropy衡量损失，得到我们的objective function

- minimize *J* = − log *P*(*w**c*|*w**c*−*m*, . . . , *w**c*−1, *w**c*+1, . . . , *w**c*+*m*)
- 由于我们的真实y是1，所以上面的公式其实是省略了log前面的1的。后面的P就是我们模型预测出的$$y_c$$ _hat 【我打不出来y_hat..】

---------------------------------------------------------

**CBOW Model:** 

Predicting a center word from the surrounding context 

For each word, we want to learn 2 vectors 

\- *v*: (input vector) when the word is in the context 

\- *u*: (output vector) when the word is in the center 

-------------------------------------------------------



##### 4.3 Skip-Gram Model

Predicting surrounding context words given a center word。和4.2 类似，我们就是将x和y交换了。

关键在于构建objective function的时候，我们假设输出y之间是互相独立的。

和4.2 的区别在于，最后的y和y_pred 一个是两个one-hot的向量，y只有一个1，但是这里的y会有多个1， 当然y_pred就是预测的probability结果



##### 4. Neagtive Sampling

由于对objective function的巨大计算开销，我们仅仅approximate结果就可以了。O（|V|），但是V都很大。

**Loss functions *J* for CBOW and Skip-Gram are expensive to compute because of the softmax normalization, where we sum over all |*V*| scores!**

更新，Objective function， gradients 以及update rule可以加快速度。

BTW，这里之所以直接使用sigmoid作为objective公式来对w和c是否来自同一个corpus是由于sigmoid的特性，所以经常被用作二分类的概率建模。

接下来给出了objective function的更新，主要是添加了sigmoid函数。

##### 4.5 Hierarchical Softmax

这个是对normal softmax的改进。复杂度为log（v），快了很多。

### w2v tutorial

link：

http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

内容要点：

1. word vector是NN task的副产品，第一层的weight就是word的vector。至于是监督学习还是非监督学习，这里不需要纠结。如果把vector看作目标，确实是非监督，但是作为task的副产品的时候，要取决于task的y到底是啥，给没有给来确定。而这个task，作者叫做“fake task”，这个特别有意思。
2. 作者自己构造了fake task，而task的构造，会对我们的后续工作以及成果造成很大的影响
3. MODEL：we can't feed a word just as a text string
   1. build a vocabulary of words from training dataset like 10,000 words
   2. One-hot vector to represent an input word
   3. output prob of a word which is around the input word given input word
   4. (1 * n) x n * m = 1 * m then the matrix n * m is like a lookup table to find the word vector [唯一不是0的n的那一位对应的每个数字会被输出生成一个vector]
   5. given word的前后word并不影响，只是学习他们是否同时出现而已



### w2v tutorial 2

延伸材料，fr 1

在1 的w2v 里面有介绍，因为原始的方法训练起来太慢了，所以作者提出了新的方法进行训练。链接如下：

link：http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/



### softmax

关于softmax的介绍，链接如下：

http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/





### 延伸：将word2vec应用在别的领域（推荐和广告）

link：

http://mccormickml.com/2018/06/15/applying-word2vec-to-recommenders-and-advertising/





论文解读：

1. [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781.pdf)(original word2vec paper) 

   link：

2. [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) (negative sampling paper)

   link：

### Personal thoughts

1. Jaccard, Cosine, Euclidean
2. SVD: V * V = (V * k) x (k * k) x (k * V)
3. Cross-entropy
4. stochastic gradient
5. binary Huffman tree