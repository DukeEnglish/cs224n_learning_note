1. hierarchical softmax，替换普通的softmax，速度更快（O(K) -> logO(K)）

2. Negative sampling 是一种update weights的方法，可以加速训练，并且效果更好。