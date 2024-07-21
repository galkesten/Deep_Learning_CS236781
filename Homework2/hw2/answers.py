r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**
1.

A. Since the Linear layer function is $XW^T+b =Z$  and the shape of $X$ is (64, 1024), We get that the shape of $W$
(512, 1024) and the shape of the output $Y$ is (64,512).
The Jacobian tensor $\frac{\partial Y}{\partial X}$ captures how each element of $Y$ changes with respect to each element of $X$.
Therefore the shape is (64, 512, 64, 1024).

B. We have that $Y_{ij}= \Sigma_{k=1}^{1024} =X_{ik}W^T_{kj} = X_{ik}W_{jk}$. 

Thus, $$\frac{\partial Y_{ij}}{\partial{X_{tl}}} = 
 \begin{cases} 
    W_{j l} & \text{if } t=i \\
    0 & \text{else}
\end{cases}
 $$
The Jacobian is a 4d Tensor such that $J[i, j] =\frac{\partial Y_{ij} }{\partial{X}}$ 
where $\frac{\partial Y_{ij}}{\partial{X}} \in M^{64, 1024}$. For each $\frac{\partial Y_{ij}}{\partial{X}}$ only the 
i-th row has non-zero elements, meaning only 1024 elements might be non-zero. This occurs for each of the 64*512 matrices
(for each $Y_{ij}$). 

Therefore, in every such matrix, only one row is non-zero, making the Jacobian tensor indeed sparse.

C. No, we can compute the partial gradient w.r.t L without calculating the jacobian tensor. 
$$
\frac{\partial L}{\partial{X}} = \Sigma_{i, j}{\frac{\partial L}{\partial{Y_{ij}}} \cdot \frac{\partial 
Y_{ij}}{\partial{X}} } = $$
$$
\Sigma_{i, j}{\frac{\partial L}{\partial{Y_{ij}}}}
\begin{pmatrix}
0 & 0 & \cdots & 0 & \cdots & 0 \\
0 & 0 & \cdots & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
W_{j1} & W_{j2} & \cdots & W_{jk} & \cdots & W_{j1024} \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0 & \cdots & 0 \\
\end{pmatrix}
=
\Sigma_{i, j}
\begin{pmatrix}
0 & 0 & \cdots & 0 & \cdots & 0 \\
0 & 0 & \cdots & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
\frac{\partial L}{\partial{Y_{ij}}} W_{j1} & \frac{\partial L}{\partial{Y_{ij}}} W_{j2} & \cdots & \frac{\partial L}{\partial{Y_{ij}}} W_{jk} & \cdots & \frac{\partial L}{\partial{Y_{ij}}} W_{j1024} \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0 & \cdots & 0 \\
\end{pmatrix}
$$

$$
=
\Sigma_{i=1}^{64} \Sigma_{j=1}^{512}
\begin{pmatrix}
0 & 0 & \cdots & 0 & \cdots & 0 \\
0 & 0 & \cdots & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
\frac{\partial L}{\partial{Y_{ij}}} W_{j1} & \frac{\partial L}{\partial{Y_{ij}}} W_{j2} & \cdots & \frac{\partial L}{\partial{Y_{ij}}} W_{jk} & \cdots & \frac{\partial L}{\partial{Y_{ij}}} W_{j1024} \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0 & \cdots & 0 \\
\end{pmatrix}
$$
$$
=
\Sigma_{i=1}^{64}
\begin{pmatrix}
0 & 0 & \cdots & 0 & \cdots & 0 \\
0 & 0 & \cdots & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
\Sigma_{j=1}^{512}\frac{\partial L}{\partial{Y_{ij}}} W_{j1} & \Sigma_{j=1}^{512}\frac{\partial L}{\partial{Y_{ij}}} W_{j2} & \cdots & \Sigma_{j=1}^{512}\frac{\partial L}{\partial{Y_{ij}}} W_{jk} & \cdots &\Sigma_{j=1}^{512} \frac{\partial L}{\partial{Y_{ij}}} W_{j1024} \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0 & \cdots & 0 \\
\end{pmatrix}
$$
$$
=
\begin{pmatrix}
\Sigma_{j=1}^{512}\frac{\partial L}{\partial{Y_{1j}}} W_{j1} & \Sigma_{j=1}^{512}\frac{\partial L}{\partial{Y_{1j}}} W_{j2} & \cdots & \Sigma_{j=1}^{512}\frac{\partial L}{\partial{Y_{1j}}} W_{jk} & \cdots &\Sigma_{j=1}^{512} \frac{\partial L}{\partial{Y_{1j}}} W_{j1024} \\
\Sigma_{j=1}^{512}\frac{\partial L}{\partial{Y_{2j}}} W_{j1} & \Sigma_{j=1}^{512}\frac{\partial L}{\partial{Y_{2j}}} W_{j2} & \cdots & \Sigma_{j=1}^{512}\frac{\partial L}{\partial{Y_{2j}}} W_{jk} & \cdots & \Sigma_{j=1}^{512}\frac{\partial L}{\partial{Y_{2ij}}} W_{j1024} \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
\Sigma_{j=1}^{512}\frac{\partial L}{\partial{Y_{ij}}} W_{j1} & \Sigma_{j=1}^{512}\frac{\partial L}{\partial{Y_{ij}}} W_{j2} & \cdots & \Sigma_{j=1}^{512}\frac{\partial L}{\partial{Y_{ij}}} W_{jk} & \cdots &\Sigma_{j=1}^{512} \frac{\partial L}{\partial{Y_{ij}}} W_{j1024} \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
\Sigma_{j=1}^{512}\frac{\partial L}{\partial{Y_{64j}}} W_{j1} & \Sigma_{j=1}^{512}\frac{\partial L}{\partial{Y_{64j}}} W_{j2} & \cdots & \Sigma_{j=1}^{512}\frac{\partial L}{\partial{Y_{64j}}} W_{jk} & \cdots &\Sigma_{j=1}^{512} \frac{\partial L}{\partial{Y_{64j}}} W_{j1024} \\
\end{pmatrix}
= 
\frac{\partial L}{\partial{Y}} \cdot W =\delta\mat{Y}W
$$

2.

A. Since the Linear layer function is $XW^T+b =Z$  and the shape of $X$ is (64, 1024), We get that the shape of $W$
is (512, 1024) and the shape of the output $Y$ is (64,512).
The Jacobian tensor $\frac{\partial Y}{\partial X}$ captures how each element of $Y$ changes with respect to each element of $W$.
Therefore the shape is (64, 512, 512, 1024).

B. We have that $Y_{ij}= \Sigma_{k=1}^{1024}X_{ik}W^T_{kj} = \Sigma_{k=1}^{1024}X_{ik}W_{jk}$. 

Thus, $$\frac{\partial Y_{ij}}{\partial{W_{tl}}} = 
 \begin{cases} 
    X_{i l} & \text{if } t=j \\
    0 & \text{else}
\end{cases}
 $$
The Jacobian is a 4d Tensor such that $J[i, j] =\frac{\partial Y_{ij} }{\partial{W}}$ 
where $\frac{\partial Y_{ij}}{\partial{W}} \in M^{512, 1024}$. For each $\frac{\partial Y_{ij}}{\partial{W}}$ only the 
j-th row has non-zero elements, meaning only 1024 elements might be non-zero. This occurs for each of the 64*512 matrices
(for each $Y_{ij}$). 

Therefore, in every such matrix, only one row is non-zero, making the Jacobian tensor indeed sparse.

C. No, we can compute the partial gradient w.r.t L without calculating the jacobian tensor. 
$$
\frac{\partial L}{\partial{W}} = \Sigma_{i, j}{\frac{\partial L}{\partial{Y_{ij}}} \cdot \frac{\partial 
Y_{ij}}{\partial{W}} } = $$
$$
\Sigma_{i, j}{\frac{\partial L}{\partial{Y_{ij}}}}
\begin{pmatrix}
0 & 0 & \cdots & 0 & \cdots & 0 \\
0 & 0 & \cdots & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
X_{i1} & X_{i2} & \cdots & X_{ik} & \cdots & X_{i1024} \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0 & \cdots & 0 \\
\end{pmatrix}
=
\Sigma_{i, j}
\begin{pmatrix}
0 & 0 & \cdots & 0 & \cdots & 0 \\
0 & 0 & \cdots & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
\frac{\partial L}{\partial{Y_{ij}}} X_{i1} & \frac{\partial L}{\partial{Y_{ij}}} X_{i2} & \cdots & \frac{\partial L}{\partial{Y_{ij}}} X_{ik} & \cdots & \frac{\partial L}{\partial{Y_{ij}}} X_{i1024} \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0 & \cdots & 0 \\
\end{pmatrix}
$$

$$
=
\Sigma_{j=1}^{512} \Sigma_{i=1}^{64} 
\begin{pmatrix}
0 & 0 & \cdots & 0 & \cdots & 0 \\
0 & 0 & \cdots & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
\frac{\partial L}{\partial{Y_{ij}}} X_{i1} & \frac{\partial L}{\partial{Y_{ij}}} X_{i2} & \cdots & \frac{\partial L}{\partial{Y_{ij}}} X_{ik} & \cdots & \frac{\partial L}{\partial{Y_{ij}}} X_{i1024} \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0 & \cdots & 0 \\
\end{pmatrix}
$$
$$
=
\Sigma_{j=1}^{512}
\begin{pmatrix}
0 & 0 & \cdots & 0 & \cdots & 0 \\
0 & 0 & \cdots & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
\Sigma_{i=1}^{64}\frac{\partial L}{\partial{Y_{ij}}} X_{i1} & \Sigma_{i=1}^{64}\frac{\partial L}{\partial{Y_{ij}}} X_{i2} & \cdots & \Sigma_{i=1}^{64}\frac{\partial L}{\partial{Y_{ij}}} X_{ik} & \cdots & \Sigma_{i=1}^{64}\frac{\partial L}{\partial{Y_{ij}}} X_{i1024} \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0 & \cdots & 0 \\
\end{pmatrix}
$$
$$
=
\begin{pmatrix}
\Sigma_{i=1}^{64}\frac{\partial L}{\partial{Y_{i1}}} X_{i1} & \Sigma_{1=1}^{64}\frac{\partial L}{\partial{Y_{i1}}} X_{i2} & \cdots & \Sigma_{i=1}^{64}\frac{\partial L}{\partial{Y_{i1}}} X_{ik} & \Sigma_{i=1}^{64}\cdots & \frac{\partial L}{\partial{Y_{i1}}} X_{i1024} \\
\Sigma_{i=1}^{64}\frac{\partial L}{\partial{Y_{i2}}} X_{i1} & \Sigma_{1=1}^{64}\frac{\partial L}{\partial{Y_{i2}}} X_{i2} & \cdots & \Sigma_{i=1}^{64}\frac{\partial L}{\partial{Y_{i2}}} X_{ik} & \Sigma_{i=1}^{64}\cdots & \frac{\partial L}{\partial{Y_{i2}}} X_{i1024} \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
\Sigma_{i=1}^{64}\frac{\partial L}{\partial{Y_{ik}}} X_{i1} & \Sigma_{1=1}^{64}\frac{\partial L}{\partial{Y_{ik}}} X_{i2} & \cdots & \Sigma_{i=1}^{64}\frac{\partial L}{\partial{Y_{ik}}} X_{ik} & \Sigma_{i=1}^{64}\cdots & \frac{\partial L}{\partial{Y_{ik}}} X_{i1024} \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
\Sigma_{i=1}^{64}\frac{\partial L}{\partial{Y_{i1024}}} X_{i1} & \Sigma_{1=1}^{64}\frac{\partial L}{\partial{Y_{i1024}}} X_{i2} & \cdots & \Sigma_{i=1}^{64}\frac{\partial L}{\partial{Y_{i1024}}} X_{ik} & \Sigma_{i=1}^{64}\cdots & \frac{\partial L}{\partial{Y_{i1024}}} X_{i1024} \\
\end{pmatrix}
= 
{\frac{\partial L}{\partial{Y}}}^T X = (\delta\mat{Y})^TX
$$

"""

part1_q2 = r"""
**Your answer:**
It is theoretically possible to compute gradients without backpropagation. Alternative methods, 
such as finite differences and forward mode automatic differentiation (AD), can also optimize neural networks. 
However, these methods are generally much less efficient and practical compared to backpropagation, 
especially for large networks and when the number of outputs is small.

Therefore, while not absolutely required, backpropagation is the preferred method for training neural networks. 
It efficiently and accurately computes the gradients of the loss function with respect to all the weights in the network. 
By applying the chain rule of calculus, backpropagation propagates the error backward through the network, layer by layer,
 allowing for effective and scalable training of deep networks. 
 Without backpropagation, calculating these gradients would be computationally infeasible and time-consuming for
  large networks.

(Finite differences is a numerical method to approximate the gradient of the loss function with respect to the weights. 
It involves perturbing each weight slightly and observing the change in the loss function.)
"""
# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    # the weights Initialize in linear layer to zero-mean gaussian noise with a standard(wstd)
    wstd = 0.1
    lr = 0.01
    reg = 0.0
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (0, 0, 0, 0, 0,)

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.05
    lr_vanilla = 0.04
    lr_momentum = 0.00418
    lr_rmsprop = 0.00016
    reg = 0.002
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum, lr_rmsprop=lr_rmsprop, reg=reg,)


def part2_dropout_hp():
    wstd, lr, = (0, 0,)
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**

1.Without dropout, the model quickly achieves high training accuracy and low training loss because it can utilize all neurons during training, 
allowing it to closely fit the training data, which matches our expectations. However, during testing, the loss is significantly higher and accuracy is lower,
as expected, because the model overfits to the training data by relying heavily on all neurons, resulting in poor generalization to new data.

With dropout, the training loss is higher and the training accuracy is lower compared to without dropout. 
However, the test loss is significantly lower and the test accuracy improves slightly, as expected. 
This is because dropout disables a fraction of neurons during training, making it harder for the model to rely on specific neurons. 
This helps address overfitting by forcing the model to generalize better and not depend too heavily on particular neurons and weights.


2.The graphs show that using a dropout rate of 0.4 led to better results in terms of both loss and test accuracy during
 training and testing compared to using a higher dropout rate.

During training with a high dropout rate, the model underfits the data, which is indicated by relatively high loss and 
low accuracy. This underfitting continues during testing, where the loss remains high and accuracy is low, 
suggesting the model struggles to fit the training data properly and doesn't generalize well, even though it performs slightly better during testing than with no dropout at all.

The reason for this is that a too-high dropout rate can be problematic because the model relies on too few neurons 
each time, making it difficult for effective learning. This decreases the model's capacity and prevents it from
learning complex patterns, leading to the poor performance observed in the graphs.

Therefore, we conclude that dropout is a useful technique to improve the generalization of deep models. 
However, it's important to balance the dropout rate to ensure the model can still learn effectively.
"""

part2_q2 = r"""
**Your answer:**
Yes, it is possible for the test loss to increase for a few epochs while the test accuracy also increases. 
This can occur because test accuracy only measures the proportion of correct predictions, 
while cross-entropy loss accounts for the confidence of those predictions.
 If the model starts making correct predictions with lower confidence or incorrect predictions with higher confidence, 
 the accuracy can improve even as the loss increases.


We can observe this phenomenon in our graph. For example, with dropout=0.4, it occurs between iterations 6-7,
 and with dropout=0, it occurs at iteration 15. Although these specific instances may vary in future runs, 
 this example demonstrates that it can indeed happen.
 
**example :** 
Consider a binary classification problem where the true labels are 0 and 1.

*Epoch 1:*

Predictions for 4 samples: $\hat{y}$= [0.4,0.8,0.6,0.3] , True labels: [0,1,1,0]

Cross-Entropy Loss Calculation: $L_{CE} = -\frac{1}{4}(log(0.6)+log(0.8)+log(0.6)+log(0.7))≈0.437$

Accuracy:  $3/4$

*Epoch 2:*

Predictions for 4 samples: $\hat{y}$= [0.45,0.9,0.55,0.4], True labels: [0,1,1,0]

Cross-Entropy Loss Calculation: $L_{CE} = -\frac{1}{4}(log(0.55)+log(0.9)+log(0.55)+log(0.6))≈0.479$

Accuracy: $4/4$

This demonstrates how the test loss can increase while the test accuracy also increases.

"""

part2_q3 = r"""
**Your answer:**


1. **Gradient Descent**: An optimization technique that aims to minimize a loss function by
iteratively adjusting the model's parameters. It works by moving in the direction of the steepest descent 
to find a local minimum.

    **Back-Propagation**: A method for efficiently computing the gradients of the loss function with 
    respect to the model's parameters. The algorithm is based on the chain rule from calculus.
    Back-propagation is typically used in optimization algorithms that require gradients.

2. Gradient Descent (GD) and Stochastic Gradient Descent (SGD) are both optimization techniques used to minimize 
loss functions by updating model parameters iteratively. 
In GD, the update rule involves computing the gradient of the loss function with respect to all training data. 
In contrast, SGD updates the model parameters using the gradient computed from a single randomly selected training example
at each iteration. The expectation of the noisy gradient updates in SGD is equal to the true gradient in GD,
allowing SGD to approximate the true gradient direction over many iterations. 
GD usually has smoother convergence compared to regular SGD, 
which can be noisy due to the variability introduced by random sampling. 

3. Stochastic Gradient Descent (SGD) is widely used in deep learning for several reasons:

    -Gradient Descent (GD) has significant memory and computation demands 
    since it requires processing the entire dataset for a single optimization step. 
    This makes it impractical for large datasets.
    
    -In Stochastic Gradient Descent (SGD), the error surface is dynamic, 
    changing with each batch of training samples. This variability can enhance 
    optimization by helping the optimizer escape flat regions or sharp local minima, 
    as these problematic features may be smoothed out in the loss surface of subsequent batches.
    
    -SGD introduces noise due to its random sampling, 
    which acts as a form of regularization. This noise can prevent the optimizer from converging to a minimum that 
    perfectly matches the training data, thereby reducing the risk of overfitting and improving 
    generalization to unseen data.
    
4.
4.A. 
The methods have the same loss output after going through entire dataset:
Mathematical Justification"

Let the dataset be $\mathcal{S}$ with size $N$. 

Split $\mathcal{S}$ into $M$ disjoint batches $\{\mathcal{B}_1, \mathcal{B}_2, \ldots, \mathcal{B}_M\}$ where 
each batch $\mathcal{B}_j$ contains $N_j$ samples such that $\sum_{j=1}^{M} N_j = N$.

The total loss over the entire dataset is:

$$
L(\theta) = \sum_{i=1}^{N} \ell(f_\theta(x_i), y_i)
$$

When split into batches, the total loss can be written as the sum of losses for each batch:

$$
L_B(\theta) = \sum_{j=1}^{M} \sum_{i \in \mathcal{B}_j} \ell(f_\theta(x_i), y_i)
$$

Since all the batches are disjoint sets, where their union is equal to entire S 
we can deduce that  $L_B(\theta) = \sum_{j=1}^{M} \sum_{i \in \mathcal{B}_j} \ell(f_\theta(x_i), y_i) =
\sum_{j=1}^{N} l(f_\theta(x_i), y_i) = L(\theta)$. 

If the losses are equivalent, then theoretically we should get the same gradient updates when 
calculating the gradient with respect to the network parameters. 
The problem is that we also need to accumulate outputs if we use the chain rule, as we will explain in the next section.

4.B.
4.A.

The methods have the same loss output after going through the entire dataset:

Let the dataset be $\mathcal{S}$ with size $N$. Split $\mathcal{S}$ into $M$ disjoint batches $\{\mathcal{B}_1, \mathcal{B}_2, \ldots, \mathcal{B}_M\}$ where each batch $\mathcal{B}_j$ contains $N_j$ samples such that $\sum_{j=1}^{M} N_j = N$.

The total loss over the entire dataset is:

$$
L(\theta) = \sum_{i=1}^{N} \ell(f_\theta(x_i), y_i)
$$

When split into batches, the total loss can be written as the sum of losses for each batch:

$$
L_B(\theta) = \sum_{j=1}^{M} \sum_{i \in \mathcal{B}_j} \ell(f_\theta(x_i), y_i)
$$

Since all the batches are disjoint sets whose union is equal to the entire dataset $\mathcal{S}$, we can deduce that:

$$
L_B(\theta) = \sum_{j=1}^{M} \sum_{i \in \mathcal{B}_j} \ell(f_\theta(x_i), y_i) = \sum_{i=1}^{N} \ell(f_\theta(x_i), y_i) = L(\theta)
$$

If the losses are equivalent, then theoretically we should get the same gradient updates when calculating the gradient
with respect to the network parameters. 
The problem is that we also need to accumulate outputs if we use the chain rule, as we will explain in the next section.

4.B.

Even though we are using batch sizes small enough to fit into memory, an out of memory error can occur because 
when we accumulate the losses from multiple forward passes, the computation graphs for each batch are also accumulated. 
This means memory usage grows with each batch processed until it exceeds the available memory.

We cannot clear the outputs because we are using the backpropagation algorithm, which requires the computational graph 
along the way. This method treats the data as if it was actually calculated together in a single forward pass, 
therefore requiring all outputs for gradient calculations. To compute the gradients correctly using the chain rule, 
we need to retain the entire computation graph until the backward pass is complete.


"""

part2_q4 = r"""
**Your answer:**

4.1. Given a computation graph where we have an edge from $f_{j-1}$ to $f_{j}$ if $f_{j-1}(a)$ is the input for $f_{j}$,
we can compute both the value of f(x0) and the value of the f'(x0) without storing
intermediate values.

The algorithm:
1. **Initialization**:
    - Set $x \leftarrow x_0$
    - Set $\text{gradient} \leftarrow 1$
    
2. **Forward Pass**:
    - For $j = 1$ to $n$:
        -  $\text{value} \leftarrow f_j(x)$
        -  $\text{gradient} \leftarrow \text{gradient} \cdot f_j'(x)$
        -  $x \leftarrow \text{value}$
    
3. **Result**:
    - The final $\text{gradient}$ is $\nabla f(x_0)$


- Memory Usage:
The algorithm uses only two variables $x$ and $\text{gradient}$ throughout the computation, 
which requires $O(1)$ memory. The computational complexity remains linear, $O(n)$.

4.2

Given a computation graph where we have an edge from $f_{j-1}$ to $f_{j}$ if $f_{j-1}(a)$ is the input for $f_{j}$,
we can compute the value of the f' without storing intermediate values. 
However, in this case we still need to store values from the first forward pass.

If each node in the computational graph store the values of the function the algorithm will be:

The algorithm:
1. **Initialization**:
    - Set $\text{gradient} \leftarrow 1$
    
2. **Forward Pass**:
    - For $j =n-1$ to $0$:
        -  $\text{gradient} \leftarrow \text{gradient} \cdot v_{j+1}.fn.derivative(v_{j}.val)$
    
3. **Result**:
    - The final $\text{gradient}$ is $\nabla f(x_0)$

The algorithm requires storing the intermediate values $v_j$ from the forward pass, leading to a memory complexity of 
$O(n)$. However we reduced memory savings by factor 2 since we don't need to save all the 'grad' properties for each vertex.
The computational complexity remains linear, $O(n)$.

4.3
In general computational graphs, memory usage is proportional to the amount of memory needed to store 
intermediate values required to evaluate a node $f_n$. 
This is because there might be multiple paths to $f_n$, necessitating the storage of intermediate values along these paths.
However, we can release memory during the calculation to optimize usage. 
This differs from the backward pass, where we need to store all intermediate values of the function valuws from 
the input node to $f_n$ in the first forward pass. 

One way to achieve ${O}(1)$ memory usage with forward mode AD theoretically is to know in advance all paths from 
$v_0$ to $v_n$ and traverse each path separately, 
then sum the contributions sequentially. However, this would require too much time due to the exponential 
number of paths in complex graphs.
We also know that when using forward mode AD we might need to perform multiple traversals for each input variable to 
compute the gradient with respect to the loss. In contrast, there are usually few outputs like a single loss function, 
so backward mode usually 
requires fewer passes on the computational graph. 
So, we usually need to trade off between memory complexity (forward AD is better) and runtime (backward AD is better).


Practical solutions that we can use for AD:
 
- **Memory Release**: During the computation, release memory for intermediate values that are no longer needed. 
This helps in managing memory more efficiently.
This helps in reducing the number of intermediate values stored.
- **Checkpointing for backward mode AD**: Store values at strategic checkpoints during the forward pass and recompute 
intermediate values as 
needed during the backward pass. This reduces the memory complexity but still requires storage of necessary values.
- **Selective Storage**: Trade off by storing only some of the intermediate values
 based on their necessity for future computations.
**Mixed Mode AD**: Use a combination of forward and backward mode AD. Apply forward mode to shallow parts of 
the network and backward mode to deeper parts where the number of parameters is larger.

4.4
Big neural networks usually have a huge number of parameters and a lot of inputs.
They usually need a lot of memory to compute the backpropagation algorithm.
Using some of the methods above can reduce the amount of memory these networks require and allow efficient learning.

"""

# ==============


# ==============
# Part 3 (MLP) answers


part3_q1 = r"""
**Your answer:**


**High  Optimization error:**

High Optimization error occurs when the training process fails to sufficiently minimize the loss function on the training data.
Neural networks often have highly non-convex loss surfaces, making it challenging to find the global optimum.
Issues such as vanishing gradients, varying rates of convergence in different dimensions, 
and dependency on initialization can contribute to high optimization error.

Solutions proposed during course:
- Advanced Optimizers: Utilize optimization algorithms such as SGD with Momentum, Adam, or RMSprop to enhance convergence rates and stability.
- Learning Rate Scheduling: Reduce the learning rate every few epoches
- Warmups: Gradually increase the learning rate at the start of training to stabilize initial updates.
- Proper Weight Initialization: Use advanced initialization methods like Xavier initialization
 to set effective starting points for the weights.
- Preprocessing Data: Normalize and preprocess data to ensure consistent gradient behavior across different axes. 
    We can also use batch normalization techniques to learn how to best normalize the data.
- Skip Connections: Introduce skip connections (e.g., as in ResNet) to mitigate vanishing gradients.

**High Generalization Error:**

High Generalization error arises when the model performs well on the training data but poorly on unseen data. 
This indicates overfitting to the training data.
This happens because we train on empirical loss instead of population loss. 

Solutions proposed during course:
- More Data: Gather more data or data that better represents the underlying distribution D
- Cross-Validation: Use cross-validation to find hyperparameters that that will not cause overfitting/underfitting.
- Regularization: Apply L1 or L2 regularization to penalize large weights.
- Early Stopping: Stop training when the validation loss starts to increase to prevent overfitting
- Mini-Batches: Use mini-batches during training to introduce some noise and prevent overfitting.
- Data Augmentation: Increase the diversity of the training data through techniques like rotation, flipping, and scaling.
- Adding Noise: Introduce noise to inputs or weights during training to regularize the model.
- Label Smoothing: Adjust labels slightly to make the model less confident and more generalizable.
- Dropout: Randomly deactivate neurons during training to prevent overfitting.
- Ensembles: Combine multiple models to improve robustness and generalization.

**High approximation Error:**

Approximation error occurs when the chosen hypothesis class H is not expressive enough to capture
the underlying patterns in the data. 

Solutions proposed during course:

- More Expressive Hypothesis Class: Use a more powerful hypothesis class, such as deep neural networks (DNNs)
- Increase Parameters: Add more layers and neurons to the network to enhance its capacity.
- Tailor the model to the specific domain, such as using convolutional neural networks (CNNs).
- Receptive Field Adjustments: In CNNs, increasing the receptive field as we go deeper in the network allows each layer
to capture features at different levels of abstraction, reducing approximation error by enabling
the model to learn more complex patterns.

"""

part3_q2 = r"""
**Your answer:**
We expect the FPR to be higher when the ratio between positive labels and negative labels in the 
training dataset does not approximate the real-life ratio. For example, in email spam detection, if we use more spam 
emails than there are in real life to expose the model to many spam samples, 
the classifier may become overly sensitive, leading to a higher FPR.

Conversely, we expect the FNR to be higher when the ratio between positive labels (e.g., disease cases) and negative 
labels (e.g., healthy cases) in the training dataset does not reflect the real-life ratio. 
This can occur in cases where positive samples are hard to obtain, such as rare diseases. 
The lack of sufficient positive examples can prevent the classifier from learning to detect them effectively,
resulting in a higher FNR.

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.101
    weight_decay = 0.0
    momentum = 0.001
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**

#### 1. Number of parameters
**Regular Block**
- **First 3x3 Convolution**:
   Parameters = $F*F*C_{in}*C_{out}+ Bias$ = (3 * 3 * 256 * 256) + 256 = 590,080
- **Second 3x3 Convolution**:
   Parameters = $F*F*C_in*C_out+ Bias$ = (3 * 3 * 256 * 256) + 256 = 590,080
- **Total Parameters for Regular Block**: 590,080 (First Convolution) + 590,080 (Second Convolution) = 1,180,160

**Bottleneck Block**

- **First 1x1 Convolution**: Parameters = (1 * 1 * 256 * 64) + 64 = 16,384 + 64 = 16,448
-  **Second 3x3 Convolution**: Parameters = (3 * 3 * 64 * 64) + 64 = 36,864 + 64 = 36,928
- **Third 1x1 Convolution**: Parameters = (1 * 1 * 64 * 256) + 256 = 16,384 + 256 = 16,640
- **Total Parameters for Bottleneck Block**: 16,448 (First Convolution) + 36,928 (Second Convolution) + 16,640 (Third Convolution) = 70,016


#### 2. Number of floating points operations:

**Regular Block**

**First 3x3 Convolution**:
- Input: A tensor of shape $(C_{in},H,W)$ = (256,H,W)
- Output: A tensor of shape $(C_{out},H,W)$ = (256,H,W). This is true because we don't change the H,W dimensions 
in residual blocks to allow the sum with shortcut the end.
- FLOPS: For each element in 1 output feature map we will have to do $F*F*C_{in}$ = $3*3*256$ operations(ignoring the addition of bias term).
We have 256 output feature maps and HW elements in each feature map so we will have to do $F*F*C_{in}*C_{out}HW = 
3*3*256*256*HW = 589824*HW$ operations 

**Second 3x3 Convolution**:

- Input: A tensor of shape $(C_{in},H,W)$ = (256,H,W)
- Output: A tensor of shape $(C_{out},H,W)$ = (256,H,W).
- Flops: Applying the same logic as before we get also  $589824*HW$ operations 

**Total Flops for regular block**: $1179648*HW$ operations

**Bottleneck Block**

**First 1x1 Convolution**:

- Input: A tensor of shape $(C_{in},H,W)$ = (256,H,W)
- Output: A tensor of shape $(C_{out},H,W)$ = (64,H,W). This is true because we don't change the H,W dimensions
- FLOPS: For each element in 1 output feature map we will have to do $F*F*C_{in} = 1*1*256$ operations(ignoring the addition of bias term).
We have 64 output feature maps and HW elements in each feature map so we 
will have to do $F*F*C_{in}**C_{out}HW$ = $1*1*256*64*HW = 16384*HW$ operations 

**Second 3x3 Convolution**:
- Input: A tensor of shape $(C_{in},H,W)$ = (64,H,W)
- Output: A tensor of shape $(C_{out},H,W)$ = (64,H,W).
- FLOPS: For each element in 1 output feature map we will have to do $F*F*C_{in} = 3*3*64$ operations. 
Therefore we get $3*3*64*64*HW = 36864HW$ operations
    
**Third 1x1 Convolution**: 
- Input: A tensor of shape $(C_{in},H,W)$ = (64,H,W)
- Output: A tensor of shape $(C_{out},H,W)$ = (256,H,W). 
- FLOPS: For each element in 1 output feature map we will have to do $F*F*C_{in} = 1*1*264$ operations(ignoring the addition of bias term).
 We have 256 output feature maps and HW elements in each feature map so we will have to do $F*F*C_{in}**C_{out}HW$ =
 $1*1*256*64*HW = 16384*HW$ operations
     
- **Total Flops for bottleneck block**: $2*16384*HW + 36864HW = 69632HW$ operations

#### 3. Ability to Combine Input

**Regular Block**

**Spatial Combination (Within Feature Maps)**:
- Each 3x3 convolution can combine information from a 3x3 neighborhood of pixels within each feature map.
- After two 3x3 convolutions, the receptive field is 5x5, meaning each output pixel can "see" a 5x5 area of the input.
- This allows for a relatively larger spatial context to be considered within each feature map.

**Feature Map Combination (Across Feature Maps)**:
- Each 3x3 convolution operates on all 256 channels and produces 256 output channels.
- Thus, we also get a strong ability to mix elements across feature maps.

**Bottleneck Block**

**Spatial Combination (Within Feature Maps)**:
- The initial 1x1 convolution does not change the spatial context—it only combines information across feature maps.
- The 3x3 convolution then combines spatial information within a 3x3 neighborhood.
- The final 1x1 convolution again does not change the spatial context.
- Overall, the receptive field for the spatial combination in a bottleneck block is 3x3, which is smaller compared to 
the regular block.

**Feature Map Combination (Across Feature Maps)**:
- **First 1x1 Convolution**:
  - This layer combines information across the feature maps (channels). It reduces the number of channels from 256 to 64. 
  Each output channel in the 1x1 convolution is a combination of all 256 input channels. 
  This allows the network to mix information from all feature maps and learn a compact representation.
- **Second 3x3 Convolution**:
  - This operates on the reduced set of channels and combines information both in the same feature map and across
  feature maps. 
- **Third 1x1 Convolution**:
  - This expands the number of channels back to the original. Again, each output channel in this 1x1 convolution 
  is a combination of all 64 input channels. This allows the network to mix information from the reduced feature maps 
  and expand it back to a richer set of features.

#### Conclusions:
We can see that the regular residual block has a stronger ability to mix elements within feature maps compared to bottleneck
blocks, and both have similar ability to combine information across feature maps. 
However, the bottleneck block has computational advantages in both FLOPs and number of parameters. 
So, we have to trade off between stronger ability to combine information and computational resources.
"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**

The results of Experiment 1.1 show that the depth of the network impacts accuracy.
The `L16` configurations (16 layers per block) with both 32 and 64 filters per layer were non-trainable, 
with test accuracy stagnant at 10%. This is likely due to vanishing gradients and overfitting, where the
network struggles
 to learn effectively. The `L8` configurations (8 layers per block) showed better performance, with `L8_K32`
 reaching approximately 60.18%
test accuracy and `L8_K64` reaching around 61.47%. Although these results were better than `L16`, 
they were not the best observed.

The `L4` configurations produced the best results. The `L4_K32` configuration achieved a
test accuracy of about 63.99%, while `L4_K64` reached around 65.51%. This works better compared to 'L2' and 'L8'.
 The `L2` configurations also performed well, particularly with more filters. 
 The `L2_K32` configuration achieved about 63.68% test accuracy, and `L2_K64` reached around 63.08%. 

In choosing hyperparameters, we focused on ensuring convergence and avoiding vanishing gradients. 
We manually tuned various hyperparameters, including learning rate and regularization. 
A very small learning rate (0.0001) was necessary for the `L8` configurations to ensure stable learning and convergence. 
We also had to use small regularization values, as larger ones would encourage
smaller weights and exacerbate the vanishing gradient problem. This 
 led to observable overfitting, as seen in the training and testing loss graphs, 
 but was necessary to prevent the gradients from vanishing entirely and to allow 
 the deeper network to learn. Despite the extensive tuning, the highest accuracy achieved
 was around 65%, indicating room for improvement.

To address the vanishing gradients problem in very deep networks,we can use 
 batch normalization which stabilize the learning process by normalizing the inputs of each layer.
  Additionally, adding residual connections (as in ResNet architectures) can provide shortcut paths for gradients, 
  allowing for more effective training of deep networks. 
These strategies can mitigate the problems associated with training very deep networks and improve their performance.
T

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""