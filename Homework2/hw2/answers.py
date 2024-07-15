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

A. The shape of $W$ is (1024,512), the shape of $X$ is (64, 1024) and the shape of the output $Y$ is (64,512).
The Jacobian tensor $\frac{\partial Y}{\partial X}$ captures how each element of $Y$ changes with respect to each element of $X$
therefore ths shape is (64, 512, 64, 1024).

B. For a given input $x_i$ the output $y_i$ is $y_i = Wx_i+b$ and the $\frac{\partial y_i}{\partial x_i} = W,
 $y_i$ is depend only on $x_i$ therefor $j\notequal i$ $y_i$ equal to zero, the result of it is the $\frac{\partial y_i}{\partial x} is sparsiy and just the diangonal with non zero.
 
 
"""

part1_q2 = r"""
**Your answer:**
Backpropagation is required to train neural networks using gradient-based optimization methods because it efficiently
 and accurately computes the gradients of the loss function with respect to all the weights in the network.
by applying the chain rule of calculus, backpropagation propagates the error backward through the network, 
layer by layer, which allows for effective and scalable training of deep networks. without backpropagation, 
calculating these gradients would be computationally infeasible and time-consuming for large networks.

However, it is theoretically possible to compute gradients without backpropagation. 
alternative methods, such as finite differences, can also optimize neural networks. 
these methods, though, are generally much less efficient and practical compared to backpropagation, especially for large networks and complex tasks. 
Therefore, while not absolutely required, backpropagation is the preferred and most effective method for training neural networks.

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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
    raise NotImplementedError()
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