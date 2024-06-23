r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**
1. False:
The in-sample error is calculated on the training dataset.
It measures how well the model fits the data it was trained on.
It can tell us how compatible the chosen hypothesis class is with the data, and it estimates the approximation error.
The test set measures something else—it checks the performance of the model on unseen data. 
Its job is to estimate the generalization error, 
which indicates if our model is able to generalize well or if there is overfitting.

2. False:
There exist splits of train-test that are better than others.
For example, if the training set contains only class A and the test set only class B,
the model will perform poorly.
We need to ensure the split is representative of the overall dataset and maintains class proportions. 
Additionally, there are issues like data leakage. 
For instance,in time series data, if the training set includes future data, the test error won't accurately estimate 
generalization error. 
Moreover, the training set should be large enough for effective learning, and the test set should be large enough for
reliable generalization error estimation.

3. True:
The purpose of cross-validation is to tune hyperparameters and evaluate the model using only the training data. 
If the test set is used during cross-validation, it leads to data leakage, 
as information from the test set would influence the training process. 
This compromises the integrity of the test set, leading to overly optimistic performance estimates 
and failing to provide an unbiased measure of the model's true performance on unseen data.

4. True:
In cross-validation, the dataset is split into multiple folds. 
In each iteration, one fold is used as the validation set while the remaining folds are used for training. 
This process is repeated for each fold, allowing the model to be trained and validated on different subsets of the data.
The performance on the validation sets across all folds helps in tuning hyperparameters and provides 
an estimate of the model's generalization error, 
indicating how well it is expected to perform on unseen data with the chosen hyperparameters in this iteration.

"""

part1_q2 = r"""
**Your answer:**
No, the friend's approach is not justified.
The test set is meant to provide an unbiased estimate of the model's performance on unseen data.
It should only be used once for final evaluation after the model has been fully trained and tuned.
By using the test set to select $\lambda$, the friend is indirectly training on the test set. This means the chosen 
$\lambda$  might work well on the test set but may not generalize to new, 
unseen data. This defeats the purpose of having a test set and can lead to overfitting.
A better approach is to use a train-validation split on the training set for hyperparameter tuning or to perform cross-validation.
This way, the test set remains a true measure of the model's generalization performance.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
The selection of $\Delta > 0$ is arbitrary because given a specific $\Delta$ and specific $\lambda$, the direction of the optimal solution to the problem will stay the same if we scale $\Delta$ to $\alpha \Delta$ and $\lambda$ to $\frac{\lambda}{\alpha}$ for $\alpha > 0$.

**Proof:**
Let's say $\mathbf{W}^*$ is an optimal solution for a problem with $\Delta > 0$ and $\lambda > 0$:
$$
L_1(\mathbf{W}^*) = \frac{1}{N} \sum_{i=1}^{N} \sum_{j \neq y_i} \max \left( 0, \Delta + \mathbf{w_j^*}^\top \mathbf{x_i} - \mathbf{w_{y_i}^*}^\top \mathbf{x_i} \right) + \frac{\lambda}{2} \| \mathbf{W}^* \|^2
$$
By the optimality condition, $L_1(\mathbf{W}^*) \leq L_1(\mathbf{W})$ for all other $\mathbf{W}$.

If we scale the inequality by $\alpha > 0$, we get $\alpha L_1(\mathbf{W}^*) \leq \alpha L_1(\mathbf{W})$ for all other $\mathbf{W}$.

Scaling $\Delta$ to $\alpha \Delta$ and $\lambda$ to $\frac{\lambda}{\alpha}$ will create the following problem:

$$
L_2(\mathbf{W}) = \frac{1}{N} \sum_{i=1}^{N} \sum_{j \neq y_i} \max \left( 0, \alpha \Delta + \mathbf{w_j}^\top \mathbf{x_i} - \mathbf{w_{y_i}}^\top \mathbf{x_i} \right) + \frac{\lambda}{2\alpha} \| \mathbf{W} \|^2
$$

Evaluating the loss $L_2$ for $\alpha \mathbf{W}^*$, we get:

$$
L_2(\alpha \mathbf{W}^*) = \frac{1}{N} \sum_{i=1}^{N} \sum_{j \neq y_i} \max \left( 0, \alpha \Delta + (\alpha \mathbf{w_j^*})^\top \mathbf{x_i} - (\alpha \mathbf{w_{y_i}^*})^\top \mathbf{x_i} \right) + \frac{\lambda}{2\alpha} \| \alpha \mathbf{W}^* \|^2
$$

$$
= \frac{1}{N} \sum_{i=1}^{N} \sum_{j \neq y_i} \alpha \max \left( 0, \Delta + \mathbf{w_j^*}^\top \mathbf{x_i} - \mathbf{w_{y_i}^*}^\top \mathbf{x_i} \right) + \frac{\lambda \alpha^2}{2\alpha} \| \mathbf{W}^* \|^2
$$

$$
= \alpha \left( \frac{1}{N} \sum_{i=1}^{N} \sum_{j \neq y_i} \max \left( 0, \Delta + \mathbf{w_j^*}^\top \mathbf{x_i} - \mathbf{w_{y_i}^*}^\top \mathbf{x_i} \right) + \frac{\lambda}{2} \| \mathbf{W}^* \|^2 \right)
$$

$$
= \alpha L_1(\mathbf{W}^*) \leq \alpha L_1(\mathbf{W}) = L_2(\alpha \mathbf{W}) = L_2(\mat{W'}) \;for\, all\, \mat{W'}
$$

The mapping $\mathbf{W}' = \alpha \mathbf{W}$ is a bijection so optimality is preserved. 
Therefore, $\alpha \mathbf{W}^*$ is an optimal solution for $L_2$.

Thus, scaling $\Delta$ and $\lambda$ in such a way preserves the optimal solution direction, indicating that what matters in the SVM problem is the ratio between $\Delta$ and $\lambda$. The choice of $\Delta$ is arbitrary as long as the ratio $\frac{\Delta}{\lambda}$ is preserved.

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

# ==============

# ==============
# Part 3 answers

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

# ==============

# ==============
