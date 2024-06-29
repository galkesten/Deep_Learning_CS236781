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
Moreover, we should split the dataset such that we have enough data for training 
(a half-half split for train and test might hurt the performance the model could achieve). On the other hand, 
choosing one example for the test set might
not be a good estimate of generalization error. So, the split needs to be balanced.

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

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
The actual number chosen for $\Delta$  is arbitrary because it does not fundamentally change the optimization problem's solution.
The value of $\Delta$ essentially sets the scale for the scores that the SVM model outputs,
a larger $\Delta$ requires a larger margin between the correct classification score 
and the scores for incorrect classifications, thus potentially driving larger values of the weight vectors 𝑊.

$\lambda$ , the regularization parameter, also affects the scale of the weight vectors but does so by penalizing their magnitude directly.
As $\lambda$ increases, the penalty for larger weight magnitudes becomes more substantial.
Therefore, adjustments to $\Delta$ and $\lambda$ have interrelated effects. Increasing $\Delta$ while decreasing $\lambda$
would potentially lead to an optimal set of weights W that are similar in nature but different in scale.

We claim that given specific set of $\lambda$, $\Delta$ and optimal $W^*$ , if we scale $\Delta$ to $\alpha \Delta$ and $\lambda$ to $\frac{\lambda}{\alpha}$ for $\alpha > 0$,  
the optimal solution of the new problem will be a scaled version of $W^*$.
 (scaling the optimal solution $W^*$ won't change the predictions of the model).


**Proof:**
Let's say $\mathbf{W}^*$ is an optimal solution for a problem with $\Delta > 0$ and $\lambda > 0$:
$$
L_1(\mathbf{W}^*) = \frac{1}{N} \sum_{i=1}^{N} \sum_{j \neq y_i} \max \left( 0, \Delta + \mathbf{w_j^*}^\top \mathbf{x_i} - \mathbf{w_{y_i}^*}^\top \mathbf{x_i} \right) + \frac{\lambda}{2} \| \mathbf{W}^* \|^2
$$
By the optimality condition, $L_1(\mathbf{W}^*) \leq L_1(\mathbf{W})$ for all other $\mathbf{W}$.

If we scale the inequality by $\alpha > 0$, we get $\alpha L_1(\mathbf{W}^*) \leq \alpha L_1(\mathbf{W})$ for all other $\mathbf{W}$.

Scaling $\Delta$ to $\alpha \Delta$ and $\lambda$ to $\frac{\lambda}{\alpha}$ will create the following problem:

$$
L_2(\mathbf{W}) = \frac{1}{N} \sum_{i=1}^{N} \sum_{j \neq y_i} \max \left( 0, \alpha \Delta + 
\mathbf{w_j}^\top \mathbf{x_i} - \mathbf{w_{y_i}}^\top \mathbf{x_i} \right) + \frac{\lambda}{2\alpha} \| \mathbf{W} \|^2
$$

Evaluating the loss $L_2$ for $\alpha \mathbf{W}^*$, we get:

$$
L_2(\alpha \mathbf{W}^*) = \frac{1}{N} \sum_{i=1}^{N} \sum_{j \neq y_i} \max \left( 0, \alpha \Delta + 
(\alpha \mathbf{w_j^*})^\top \mathbf{x_i} - (\alpha \mathbf{w_{y_i}^*})^\top \mathbf{x_i} \right) + \frac{\lambda}{2\alpha} \| \alpha \mathbf{W}^* \|^2
$$

$$
= \frac{1}{N} \sum_{i=1}^{N} \sum_{j \neq y_i} \alpha \max \left( 0, \Delta + \mathbf{w_j^*}^\top \mathbf{x_i} 
- \mathbf{w_{y_i}^*}^\top \mathbf{x_i} \right) + \frac{\lambda \alpha^2}{2\alpha} \| \mathbf{W}^* \|^2
$$

$$
= \alpha \left( \frac{1}{N} \sum_{i=1}^{N} \sum_{j \neq y_i} \max \left( 0, \Delta + \mathbf{w_j^*}^\top \mathbf{x_i} 
- \mathbf{w_{y_i}^*}^\top \mathbf{x_i} \right) + \frac{\lambda}{2} \| \mathbf{W}^* \|^2 \right)
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
From the weights visualization, we can see that the linear model has learned the general shapes of most digits. 
The areas with higher weight values correspond to where the digit's pixels are white, 
which means the model has picked up the digit shapes. The model assigns higher weights to pixels 
it thinks are more important for identifying each digit.

For digits like 0, 1, 2, and 3, the weights form clear, recognizable shapes, 
showing that the model has learned these digits well. But for digits like 4, 5, 6, 7, and 9, the weights are less clear, 
making it harder for the model to distinguish them accurately. We see the model often mistakes 4 for 7, 4 for 9, and 5 for 6. These similarities in weights and shapes lead to misclassifications,
indicating that a simple linear model struggles with these more complex shapes.

Additionally, it's important to note that the model's weights depend on the position of the digits in the image. 
If a digit is moved within the image, the model's performance might get worse. 
"""

part2_q3 = r"""
**Your answer:**
1) Based on the graph, it looks like the learning rate we chose (0.01) is pretty good. 
The loss decreases steadily during training until it converges to a minimum. 
The graph is smooth and doesn't show a lot of fluctuations.
If the learning rate was too low, we would see a very slow decrease in the loss. 
By the end of the training, we might barely see it converging to a specific value. 
The graph would just show a slow, almost linear decrease.
On the other hand, if the learning rate was too high, we would see a lot of fluctuations in the graph.
The loss values would jump up and down a lot, and we wouldn't see a clear convergence. 
This is because large gradient steps can cause the optimization process to overshoot the minimum,
leading to oscillations rather than convergence.

2) Based on the graph of the training and test set accuracy, 
it looks like the model is slightly overfitted to the training set.
The training accuracy is a bit higher than the validation accuracy, but the difference isn't big. 
This shows the model performs better on the training data but still does pretty well on the validation data.
The performance on the test set is also quite good (89%) compared to train (close to 93%), which is a good indication 
that the model learned to generalize well. 
In the case of high overfitting, we would expect that while the training accuracy keeps increasing, 
the validation accuracy would peak and then start to decrease as the model gets too good at the training data and starts 
to memorize it. Underfitting is when we see low performance even on training data, which is not the case here.

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
In a linear regression model, 
we assume that the labels have a linear relationship with the data, 
with some Gaussian noise that has a mean of zero. Because of this, the ideal residual plot should show a random scatter of residuals around zero. Most residuals should be close to zero, without any specific pattern where errors are consistently positive or negative. Additionally, we want the residuals to have the same variance across all levels of the independent variables, which is called homoscedasticity.

The residual plot for the top-5 features shows that the model's predictions 
are generally close to the true values, as most residuals are centered around zero,
with an average spread between -5 and 5. 
However, we do notice that for higher predicted values, the spread of the residuals
is higher, indicating slight heteroscedasticity. 
There are also a few outliers with large residuals, 
which the model doesn't handle well.

After adding non-linear features and tuning the hyperparameters, 
we see that the spread of residuals has become tighter and more centered around 
zero, with an average spread between -2.5 and 2.5. 
This indicates an improvement in the model's predictions, 
with fewer large errors and a more consistent variance across 
different predicted values.
"""

part3_q2 = r"""
**Your answer:**
1) Yes, this is still a linear regression model. 
Adding non-linear features means we are transforming
the original feature space into a higher-dimensional space. 
The model itself remains linear in this new feature space,
 using the same learning algorithm. 
 The difference is that we are preprocessing by creating new features, 
 but the regression performed is still linear with respect to these new features.

2)No, we can't fit just any non-linear function of the original features with this approach. 
When we do feature engineering, we make assumptions about the types of functions we're looking for. 
For example, by choosing polynomial features, 
we're specifically searching for parameters of a polynomial of a certain degree, 
not any possible function. With this approach, we must know in advance which types of functions we're looking for. 
We can't fit any arbitrary function without knowing the structure of the function we are aiming to model. 
This is unlike neural networks, which can learn the correct transformations themselves through training.


3) In a linear classification model,
the parameters define a hyperplane that represents the decision boundary.
Adding non-linear features changes this boundary in the original feature space.
While the decision boundary remains a hyperplane in the transformed feature space,
it translates to a non-linear decision boundary in the original feature space.
For example, using polynomial features can result in decision boundaries that
are hyperbolas, circles, or other polynomial shapes in the original space.
"""

part3_q3 = r"""
**Your answer:**
1) We learned in machine learning intro that the regularization parameter $\lambda$ can vary a lot. A high  $\lambda$ 
encourages solutions with smaller weights, which can help with generalization, while a low 
$\lambda$  allows the model to fit the training data better, which can lead to overfitting.
Because $\lambda$ can take on a wide range of values, 
using np.logspace is better than np.linspace. np.logspace allows us to explore both small and large 
$\lambda$ values more effectively because the gaps between numbers are on a logarithmic scale, not a linear one. 
If we used a linear scale, we would need to check a lot more hyperparameters for the same range of values.


2) We chose 20 different values for $\lambda$  and 3 different values for the polynomial degree, 
giving us $20 \times 3 = 60$ combinations. 
Each combination is evaluated using 3-fold cross-validation. 
Therefore, the model is fitted $60 \times 3 = 180$ times 
in total during the cross-validation process. 

"""

# ==============
