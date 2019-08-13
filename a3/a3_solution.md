### 1. Machine Learning & Neural Networks

Added material:

[saddle point](https://blog.csdn.net/baidu_27643275/article/details/79250537) this is Chinese version. For English, pls check on wiki

# [Challenges for mini-batch GD](http://ruder.io/optimizing-gradient-descent/index.html#fn2)

Vanilla mini-batch gradient descent, however, does not guarantee good convergence, but offers a few challenges that need to be addressed:

- Choosing a proper learning rate can be difficult. A learning rate that is too small leads to painfully slow convergence, while a learning rate that is too large can hinder convergence and cause the loss function to fluctuate around the minimum or even to diverge.
- Learning rate schedules [[1\]](http://ruder.io/optimizing-gradient-descent/index.html#fn1) try to adjust the learning rate during training by e.g. annealing, i.e. reducing the learning rate according to a pre-defined schedule or when the change in objective between epochs falls below a threshold. These schedules and thresholds, however, have to be defined in advance and are thus unable to adapt to a dataset's characteristics [[2\]](http://ruder.io/optimizing-gradient-descent/index.html#fn2).
- Additionally, the same learning rate applies to all parameter updates. If our data is sparse and our features have very different frequencies, we might not want to update all of them to the same extent, but perform a larger update for rarely occurring features.
- Another key challenge of minimizing highly non-convex error functions common for neural networks is avoiding getting trapped in their numerous suboptimal local minima. Dauphin et al. [[3\]](http://ruder.io/optimizing-gradient-descent/index.html#fn3) argue that the difficulty arises in fact not from local minima but from saddle points, i.e. points where one dimension slopes up and another slopes down. These saddle points are usually surrounded by a plateau of the same error, which makes it notoriously hard for SGD to escape, as the gradient is close to zero in all dimensions.

*Above are the main reason why we have to learn things like Adam-optimizer and also why they were argued.*

#### A. Adam optimizer

1. Q：**how using m stops the updates from varying as much and why this low variance may be helpful to learning, overall.**

   Momentum :

- The momentum term **increases** for dimensions whose **gradients point in the same directions** and reduces updates for dimensions whose gradients change directions 
  - 动量可以让和上一个timestep**方向一致**的梯度正常增加，而**不一致**的梯度的更新变小
- Gain faster convergence and reduced oscillation. 
  - 可以使模型更快收敛并且减少震荡
    - note：Essentially, when using momentum, we push a ball down a hill. The ball accumulates momentum as it rolls downhill, becoming faster and faster on the way (until it reaches its terminal velocity if there is air resistance, i.e. γ<1γ<1). The same thing happens to our parameter updates: The momentum term increases for dimensions whose gradients point in the same directions and reduces updates for dimensions whose gradients change directions. As a result, we gain faster convergence and reduced oscillation. 【I privately think this is a vivid description how momentum works】

2. Adaptive learning rate:

   从公式上我们可以看到，梯度越大的参数，相应的v越大，那么更新就越小。为什么这个策略有效呢？

- Used to **normalize** the parameter update step, element wise
- Weights that receive high gradients will have their effective learning rate reduced
- Weights that receive small / infrequent updates will have effective learning rate increased

#### B. Dropout
- $\gamma$ = 1/p for scaling output on training (Not sure)

- Dropout in training but not in evaluating :
    
    因为在评估的时候，我们想要尽可能的使用输入的信息做判断。如果在这个时候做了drop，那就相当于传递到这个神经元的信息损失了。
    
    况且在训练的时候进行drop，意义在于不学习到过多的训练数据的信息以免overfitting

### Prob 2. Neural Transition-Based dependency parsing
#### A. Parsing a sentence

| Stack                          | Buffer                                 | New dependency      | Transition | step |
| ------------------------------ | -------------------------------------- | ------------------- | ---------- | ---- |
| [ROOT]                         | [I, parsed, this, sentence, correctly] |                     | Init       | 0    |
| [ROOT, I]                      | [parsed, this, sentence, correctly]    |                     | Shift      | 1    |
| [ROOT, I, parsed]              | [this, sentence, correctly]            |                     | Shift      | 2    |
| [ROOT, parsed]                 | [this, sentence, correctly]            | I <- parsed         | Left-Arc   | 3    |
| [ROOT, parsed, this]           | [ sentence, correctly]                 |                     | Shift      | 4    |
| [ROOT, parsed, this, sentence] | [correctly]                            |                     | Shift      | 5    |
| [ROOT, parsed, sentence]       | [correctly]                            | this <- sentence    | Left-Arc   | 6    |
| [ROOT, parsed]                 | [correctly]                            | parsed -> sentence  | Right-Arc  | 7    |
| [ROOT, parsed, correctly]      | []                                     |                     | Shift      | 8    |
| [ROOT, parsed]                 | []                                     | parsed -> correctly | Right-Arc  | 9    |
| [ROOT]                         | []                                     | Root -> parsed      | Right-Arc  | 10   |

#### B. Number of steps

A sentence contain N words will be parsed in 2N steps:
 - Need total N "SHIFT" operations to read all words in a sentence
 - Each time using an "Arc" operation, one word in stack is removed
 - When finish parsing, Stack only contains ROOT
 - So total removed words in Stack is N, which mean we need N "Arc" operations
 - Total steps: N (SHIFT) + N (Arc) = 2N

#### F. Four types of parsing error examples (Need contribution):

- Prepositional Phrase Attachment Error
- Verb Phrase Attachment Error
- Modifier Attachment Error
- Coordination Attachment Error

#### EVALUATION:
After train, model has 88.46 UAS on DEV set and 88.85 UAS on Test set
#### TAKE AWAY:
- Feature extraction for Neural Dependency Parsing: 
  - Indices of top *n* words in stacks and buffers, and its children
  - Embedding of words, POS tags and dependency relations can be trained together 
- Adaptive learning rate affects the generalization of model
- Pytorch
```python
# Save and load weight
torch.save(model.state_dict(), path)
model.load_state_dict(torch.load(path))

## Drop out layer
model.train() # put this before train to enable train mode : apply drop out to model
model.eval() # don't apply drop out when evaluate

## nn.Embedding take input as vector of index, return embedding vectors
```



Reference：

1. [Optimizing gradient descent - Sebastian Ruder](http://ruder.io/optimizing-gradient-descent/index.html)
2. [CS231N - Neural networks 3](http://cs231n.github.io/neural-networks-3/)

https://github.com/Luvata/CS224N-2019/tree/master/Assignment/a3 