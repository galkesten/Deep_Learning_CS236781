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
    lr_vanilla = 0.1
    lr_momentum = 0.005
    lr_rmsprop = 0.0001
    reg = 0.005
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
1.
Without dropout, the model quickly achieves high training accuracy and low training loss because it can utilize all neurons during training, 
allowing it to closely fit the training data, which matches our expectations. However, during testing, the loss is significantly higher and accuracy is lower,
 as expected, because the model overfits to the training data by relying heavily on all neurons, resulting in poor generalization to new data.
 
With dropout, the training loss is higher and the training accuracy is lower compared to without dropout. 
However, the test loss is significantly lower and the test accuracy improves slightly, as expected. 
This is because dropout disables a fraction of neurons during training, making it harder for the model to rely on specific neurons. 
This helps address overfitting by forcing the model to generalize better and not depend too heavily on particular neurons and weights.


2.
If the dropout rate is too high, it can be problematic because the model will rely on too few neurons each time, 
making it difficult for the model to learn effectively. This can lead to poor performance as we see in the graphs.
We observe that during training, the model underfits the data as indicated by the relatively high loss and low accuracy. 
This underfitting persists during testing, where the loss remains high and the accuracy is low, 
suggesting that the model fails to fit the training data properly and does not generalize well.

Dropout is a useful technique to improve the generalization of deep models, as we see. 
However, it is important to balance the dropout rate to ensure that the model can still learn effectively.


2.
A high dropout rate may excessively eliminate neurons, impairing the model's capacity to assimilate crucial information from the data.
 This excessive neuron dropout can result in underfitting, 
 where the model is unable to detect essential patterns and exhibits inadequate performance.

"""

part2_q2 = r"""
**Your answer:**
Yes, it is possible for the test loss to increase for a few epochs while the test accuracy also increases. 
This can occur because test accuracy only measures the proportion of correct predictions, 
while cross-entropy loss accounts for the confidence of those predictions.
 If the model starts making correct predictions with lower confidence or incorrect predictions with higher confidence, 
 the accuracy can improve even as the loss increases.


We can observe this phenomenon in our graph. For example, with dropout=0.4, it occurs between iterations 6-7,
 and with dropout=0, it occurs at iteration 10. Although these specific instances may vary in future runs, 
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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q4 = r"""
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
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
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
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
    lr = 0.1
    weight_decay = 0.0
    momentum = 0.001
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
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
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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