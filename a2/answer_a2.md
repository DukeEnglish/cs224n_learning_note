##  part 1

**(a)** 

As described in the doc, $\boldsymbol{y}$ is a one-hot vector with a 1 for the true outside word $o$, that means $y_i$ is 1 if and only if $i == o$. so the proof could be below:

$\begin{aligned} - \sum_{w\in Vocab}y_w\log(\hat{y}_w) &= - [y_1\log(\hat{y}_1) + \cdots + y_o\log(\hat{y}_o) + \cdots + y_w\log(\hat{y}_w)] \ & = - y_o\log(\hat{y}_o) \ & = -\log(\hat{y}_o) \ & = -\log \mathrm{P}(O = o | C = c) \end{aligned}$

**(b)** 

$$J = Cross-Entropy(y, \hat{y}) = - \sum_{w\in Vocab}y_w\log(\hat{y}_w)$$

$$\hat{y}= softmax(\theta)$$

$$\theta=U^Tv_c$$

If you feel unclear, please add subtitle to each variable associated to c [same as v_c]
$$
\frac{\partial J}{\partial v_c} = \frac{\partial J}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial \theta} \frac{\partial \theta}{\partial v_c}
$$
$$\frac{\partial J}{\partial \hat{y}} = -\frac{y_w} {\hat{y}_w}$$

if fraction is equal u_o·v_c, please see this site for the details of the derivation of [softmax](https://blog.csdn.net/bqw18744018044/article/details/83120425)

$$\frac{\partial \hat{y}}{\partial \theta} = \hat{y}(1- \hat{y}) $$

else:

$$\frac{\partial \hat{y}}{\partial \theta} = -\hat{y}y$$

therefore: 

$$\frac{\partial J}{\partial \theta} = (\hat{y} - y) $$

$$\begin{aligned}  \frac{\partial \theta}{\partial v_c} \ \end{aligned} = \frac{\partial U^Tv_c}{\partial v_c}$$

$$ (\hat{y} - y) \frac{\partial U^Tv_c}{\partial v_c} \ = U^T(\hat{y} - y) $$

**(c)** similar to the equation above. $$\begin{aligned} \frac{\partial J}{\partial v_c} &= \frac{\partial J}{\partial \theta} \frac{\partial \theta}{\partial U} \ &= (\hat{y} - y) \frac{\partial U^Tv_c}{\partial U} \ &= v_c^T(\hat{y} - y) \end{aligned}$$

To be clear, we have $$\frac{\partial U^Tv_c}{\partial U} = \frac{\partial v_c^TU}{\partial U}$$

**(d)** $$s = 1/(1+e^{-x})$$

Details: 

$$s = 1/f(x)$$

$$f(x) = 1+ e^{g(x)}$$

$$g(x) = -x$$

so, answer is:

$$s(x)(1-s(x))$$

**(e)** I prefer the old version question

Repeat part (a) and (b) assuming we are using the negative sampling loss for the predicted vector vc, and the expected output word is o. Assume that K negative samples (words) are drawn, and they are 1, · · · , K, respectively for simplicity of notation (o∉1,…,Ko∉1,…,K). Again, for a given word, o, denote its output vector as uouo. The negative sampling loss function in this case is K

Jneg−sample(o,vc,U)=−log(σ(u⊤ovc))−K∑k=1log(σ(−u⊤kvc))Jneg−sample(o,vc,U)=−log⁡(σ(uo⊤vc))−∑k=1Klog⁡(σ(−uk⊤vc))

where σ(·) is the sigmoid function. After you’ve done this, describe with one sentence why this cost function is much more efficient to compute than the softmax-CE loss (you could provide a speed-up ratio, i.e., the runtime of the softmax- CE loss divided by the runtime of the negative sampling loss). *Note: the cost function here is the negative of what Mikolov et al had in their original paper, because we are doing a minimization instead of maximization in our code.*

u_o: 

(only one element contains u_o)

$$(s(u_o^Tv_c)-1)v_o $$

u_k:

$$-(s(-u_k^Tv_c)-1)v_o $$ for k = 1,2,3,4…K (partial derivatives for each u_k)

v_c:

$$(s(u_o^Tv_c)-1)u_o^T -(s(-u_k^Tv_c)-1)u_k$$

**answer to the efficient:** Here we just need to update num(sample)+1 weights, which save much time, comparing to the old version where we need to update all weights whose number is associated to the number of words.

**(f)**

first two is just equal to two equation mentioned in the question and the other one shows below:

(iii) 0

## part 2

refer to code file



reference:

1. https://github.com/ankit-ai/cs224n-natural-language-processing-winter2019
2. https://github.com/ZacBi/CS224n-2019-solutions
3. https://blog.csdn.net/bqw18744018044/article/details/83120425
4. https://github.com/Observerspy/CS224n/blob/master/assignment2/assignment2-soln.pdf
5. 



