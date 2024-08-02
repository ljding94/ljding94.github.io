# Machine Learning - Andrew Ng

## Introduction

### Introduction

### Model and Cost Function

### Parameter Learning
- Gradient Descent
    - $\theta_j = \theta_j - \alpha \frac{\partial}{\partial \theta_j}J(\theta_0,\theta_1)$
    - $\alpha$:learning rate
    - $J(\theta_0,\theta_1)$: cost function
- Gradient Descent Intuition
- Gradient Descent for linear regression

## Linear regression with multiple variables
### Multivariate linear regression
- multiple features
- gradient descent for multiple variables
- gradient descent: feature scaling
    - mean normalization
- gradient descent: learning rate
    - debugging: plot $J(\theta)$ vs. # of iterations
    - choosing learning rate:
- features and polynomial regression

### Computing parameters analytically
- Normal equation $X^TX\theta=X^Ty$
    - can be slow if number of feature is large
- Normal equation and non-invertibility
    - redundant features (linearly dependent)
    - too many features ($m\leq n$)

## Logistic regression
### Classification and representation
- classification: $y=0$ or $y=1$
- hypothesis representation: $h_\theta(x)=g(\theta^Tx)=1/(1+e^{-\theta^Tx})$, sigmoid function
     interpretation of $h$: estimated probability of $y=1$
- decision boundary:
    - say $h_\theta(x)=g(\theta^Tx)$, $\theta^Tx$ gives the decision boundary
    - non-linear decision boundary: polynomial in $g(\sim x)$, e.g. $g(\theta_1x_1^2+\theta_2x_2^2)$

### Logistic regression model
- cost function $$J(\theta) = \frac{1}{m}\sum_{i=1}^{m}Cost(h_\theta(x^{(i)}),y^{(i)})$$
    - Linear regression: $$Cost(h_\theta(x^{(i)}),y^{(i)})=\frac{1}{2}(h_\theta(x^{(i)})-y^{(i)})^2$$
    - Logistic regression: $$Cost(h_\theta(x),y)=\begin{cases} -log(h_\theta(x)) & y=1 \\-log(1-h_\theta(x)) & y=0\end{cases}$$

- simplified cost function and gradient descent
    - $Cost(h_\theta(x),y)=-ylog(h_\theta(x))-(1-y)log(1-h_\theta(x))$
    - $J(\theta) = \frac{1}{m}\sum_{i=1}^{m}Cost(h_\theta(x^{(i)}),y^{(i)})$
    - gradient descent $\frac{\partial}{\partial\theta_j}J(\theta) = \frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j$
    - note: same form as linear regression expect for $h_\theta(x)$

- Advanced optimization
    - Goal: minimize cost function $J(\theta)$
    - algorithms:
        - gradient descent
        - conjugate gradient
        - BFGS
        - L-BFGS

### Multiclass classification
- One-vs-all (one-vs-rest): multiple classifier

### Solve the problem of overfitting
- overfitting: too many feature, hypothesis may fit too well
- address overfitting:
    - 1. reduce number of features
    - 2. regularization: keep all the features but reduce magnitude/values of $\theta$
- cost function: $$J(\theta)=\frac{1}{m}\sum_{i=1}^{m}Cost(h_\theta(x^{(i)}),y^{(i)})+\lambda\sum_{j=1}^{n}\theta_j^2$$
- Regularized linear regression
    - gradient descent: $\frac{\partial}{\partial\theta_j}J(\theta) = \frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j+\frac{\lambda}{m}\theta_j$
    - normal equation: $(X^TX+\lambda L)\theta=X^Ty$, $L=diag(0,1,...,1)$
- Regularized logistic regression
    - gradient descent (same form)

## Neural Networks: Representation
### Motivation
- Non-linear hypotheses
- Neurons and the brain: part of brain for touch can learn how to see
### Neural networks
- Model representation
- forward propagation
    - $z^{2} = \Theta^{(1)}x$
    - $a^{(2)} = g(z^{(2)})$
    - add a_0^{(2)} = 1
    - $z^{(3)}= \Theta^{(2)}a^{(2)}$
    - $h_\theta(x) = a^{(3)} = g(z^{3})$
### Application
- examples and intuitions
    - non-linear classification: XOR/XNOR
- multiclass classification

## Neural Networks: Learning
### Cost function and backpropagation
- Cost function:
    - Logistic regression with regulation: $J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}log(h_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta(x^{(i)}))\right]+\frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2$
    - Neural network: $J(\Theta) = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^K\left[-y^{(i)}_klog(h_\Theta(x^{(i)})_k)+(1-y^{(i)}_k)log(1-h_\Theta(x^{(i)})_k)\right]+\frac{\lambda}{2m}\sum_{l=1}^{L-1}\sum_{i=1}^{s_l}\sum_{j=1}^{s_{l+1}}(\Theta_{j,i}^{l})^2$
- gradient computation: backpropagation algorithm
    - intuition: $\delta_j^{(l)}=$ "error" of node $j$ in lager $l$.
    - For each output unit (layer $L=4$):
        - $\delta_j^{(4)} = a_j^{(4)}-y_i$
        - $\delta_j^{(3)} = (\Theta^{(3)})^T\delta^{(4)}g'(z^{(3)})$
        - $\delta_j^{(2)} = (\Theta^{(2)})^T\delta^{(3)}g'(z^{(2)})$
- backpropagation intuition

### Backpropagation in practice
- implementation note: unrolling parameters:
    - have initial parameters $\Theta^{(1)},\Theta^{(2)},\Theta^{(3)}$
    - unroll to get initial $\Theta$ to pass to function
- gradient checking
    - check numerical estimate of gradient
    - $\frac{\partial}{\partial\theta}J(\theta) \sim \frac{J(\theta+\epsilon)-J(\theta-\epsilon)}{2\epsilon}$
    - $\epsilon \sim 10^{-4}$
    - compare computed gradient against numerical gradient
    - make sure disable checking code before training classifier (backpropagation)
- random initialization
- putting it together


### Application of neural networks
- autonomous driving


## Advice for applying machine learning

### Evaluating a learning algorithm
- deciding what to try next
    - possible choices
        - get more training examples
        - try smaller sets of features
        - try getting additional features
        - try adding polynomial features
        - try decreasing $\lambda$
        - try increasing $\lambda$
    - machine learning diagnostic
- evaluating a hypothesis
    - training set and test set
        - learn $\theta$ from training data
        - compute test set error
        - misclassification error (0/1 error)
- modeling selection and train/validation/test set
    - model selection itself is another fitting parameter
    - training + cross validation + test ~ 60%+20%+20%
    - training error(parameter) + cross validation error(model selection parameter) + test error(generalization error)
### Bias vs. variance
- diagnosing bias vs variance
    - bias -> high training error + high validation error
    - variance -> low training error + high validation error

- regularization and bias/variance

- learning curves
    - J vs. m (training set size)
    - bias -> J levels off, getting more data not helping
    - variance -> large gap between J train and J validation: more data fo helps

- deciding what do next revisited
    - high variance
        - more training examples
        - smaller sets of features
        - increasing $\lambda$
    - high bias
        - additional features
        - adding polynomial features
        - decreasing $\lambda$



## Support vector machines

### Large margin classification
    - optimization objective

### kernels

### SVMs in Proctice






