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

3.Answer:


4.Answer:

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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
