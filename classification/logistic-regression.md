
# Logistic Regression

## Logistic Regression Assumptions

These assumptions are often written in the equivalent forms:

$\begin{array}{ll}P(Y=1 \mid \mathbf{X}=\mathbf{x})=\sigma\left(\theta^{T} \mathbf{x}\right) & \text { where we always set } x_{0} \text { to be } 1 \newline P(Y=0 \mid \mathbf{X}=\mathbf{x})=1-\sigma\left(\theta^{T} \mathbf{x}\right) & \text { by total law of probability }\end{array}$

**This probability, ranging from 0 to 1, can be used as a criterion to classify the new observation. The higher the value of p, the more likely the new observation belongs to class y = 1, instead of y = 0.**


## Logaritmic transformation of the sigmoid function

Recall that $P(Y|X)$ (or $p(x)$) can be approximated as a sigmoid
function applied to a linear combination of input features.

### From sigmoid function to **logit function**

$p(x) =  \frac{e^x}{1 + e^x}$


$p(x) =  \frac{e^{f(x)}}{1 + e^{f(x)}}$


where

$$f(x) = \theta_0 + \theta_1 x$$

$p(x) =  \frac{e^{\theta_0 + \theta_1x}}{1 + e^{\theta_0 + \theta_1 x}}$


$p(1 +  e^{\theta_0 + \theta_1 x}) =  e^{\theta_0 + \theta_1x}$


$p  + pe^{\theta_0 + \theta_1 x} =  e^{\theta_0 + \theta_1x}$


$p  =  e^{\theta_0 + \theta_1x} -  pe^{\theta_0 + \theta_1 x}$


$p  =  e^{\theta_0 + \theta_1x}(1 -  p)$

$\frac{p}{1-p}  =  e^{\theta_0 + \theta_1x}$

then we have:

$$log\left(\frac{p}{1-p}\right)  = \theta_0 + \theta_1x$$


It follows the case for:

$$
\begin{aligned}
&log \left(\frac{p}{1-p}\right)= \theta_{0} + \theta_{1} x_{1}+ \theta_{2} x_{2} + \ldots + \theta_{m} x_{m}
\end{aligned}
$$

### From logit function to sigmoid function

The standard logistic function is simply the inverse of the logit equation .
If we solve for p from the logit equation.


$log(\frac{p}{(1-p)}) = \theta_0 + \theta_1x$


$\frac{p}{(1-p)} = e^{\theta_0 + \theta_1x}$

$y_i = e^{\theta_0 + \theta_1x}$


$p = e^{\theta_0 + \theta_1x} (1-p)$


$p = e^{\theta_0 + \theta_1x} - pe^{\theta_0 + \theta_1x}$


$p + pe^{\theta_0 + \theta_1x} = e^{\theta_0 + \theta_1x}$


$p (1 + e^{\theta_0 + \theta_1x}) = e^{\theta_0 + \theta_1x}$

and then, we have the model for **Simple Logistic Regression**:

$$p(x) = \frac{e^{\theta_0 + \theta_1x}}{1 + e^{\theta_0 + \theta_1x}}$$

<!---
|![Alt Text](https://miro.medium.com/max/1400/1*XRCJt-5yNXDfzrVbEbh4DA.gif)|
|:--:|
|Gradient Descent|
|[Source link](https://towardsdatascience.com/animations-of-logistic-regression-with-python-31f8c9cb420)|
-->

It follows the case for **Multiple Logistic Regression**:

$$p(x) = \frac{e^{\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots \theta_m x_m }}{1 + e^{\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots \theta_m x_m }} = \frac{1}{1+e^{-\left(\theta_0 + \theta_{1}  x_{1} + \theta_{2}  x_{2}+\ldots + \theta_{m}  x_{m-1}\right)}}
$$

<!---
|![Alt Text](https://miro.medium.com/max/1152/1*IsOs8KXweJy4uFcSRc45RQ.gif)|
|:--:|
|Gradient Descent|
|[Source link](https://towardsdatascience.com/animations-of-logistic-regression-with-python-31f8c9cb420)|
-->

## The Loss Function

Like linear regression, the logistic regression algorithm finds the best values of coefficients to fit the training dataset. How do we find the best fit model?
Can we use the same estimation method, Ordinary Least Squares (OLS), as linear regression?

The answer is NO

### The Standard Ordinary Least Squares
The OLS targets minimizing the sum of squared residuals, which involves the difference of the predicted output and the actual output:

$$\sum\limits_{i=1}^n (y_i - \hat{y})^2$$

In contrast, the actual output in the logistic linear equation is:   

$$\frac{p}{1-p}$$

we can’t calculate its value since we don’t know the value of p.
The only output we know is the class of either y = 0 or 1. So we have to use another estimation method.


### Maximun Likelihood Estimation (MLE)

MLE for short, is a probabilistic framework for estimating the parameters of a model. **In MLE, we wish to maximize the conditional probability of observing the data (X)** given a specific probability distribution and its parameters ($\theta$), stated formally as:

$$P(X|\theta)$$           


Where X is, the joint probability distribution of all observations from the problem domain from 1 to n:


$$P(x_1, x_2, \ldots, x_n|\theta)$$


This resulting conditional probability is referred to as the likelihood function of observing the data given the model parameters ($\theta$) and written using the notation:

$$
L(\mathbf{\theta})=\prod_{i=1}^{N} p\left(y^{(i)} \mid \mathbf{x}^{(i)} ; \mathbf{\theta}\right)
$$

We use a likelihood function that measures how well a set of parameters ($\theta$’s) fit a sample of data.  In other words, the goal is to make inferences about the population that is most likely to have generated the training dataset. **We can frame the problem of fitting a machine learning model as the problem of probability density estimation**.

### Probabilistic Interpretation

Recall that $(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)$ are independently generated. So the probability of getting  $y_1, y_2, \ldots, y_n$  in distribution D from the corresponding is:          

$$
p\left(y_{1}, \ldots, y_{n} \mid x_{1}, \ldots, x_{n}\right)=\prod_{i=1}^{N} p\left(y^{(i)} \mid \mathbf{x}^{(i)} ; \mathbf{\theta}\right)$$                                                                           

This gives us the likelihood of the parameters for n training examples:

$$
L(\mathbf{\theta})=\prod_{i=1}^{N} p\left(y^{(i)} \mid \mathbf{x}^{(i)} ; \mathbf{\theta}\right)
$$                                                                      

However, we have two separate terms for $p\left(y=1 \mid \mathbf{x}^{(i)}; \mathbf{\theta}\right)$ and  $p\left(y=0 \mid \mathbf{x}^{(i)}; \mathbf{\theta}\right)$. Nonetheless, it is possible to combine those two terms into one like:

$$
p\left(y^{(i)} \mid \mathbf{x}^{(i)} ; \mathbf{\theta}\right)=p\left(y=1 \mid \mathbf{x}^{(i)} ; \mathbf{\theta}\right)^{y^{(i)}} p\left(y=0 \mid \mathbf{x}^{(i)} ; \mathbf{\theta}\right)^{1-y^{(i)}}
$$

The above trick is used a lot in Machine Learning. It is easy to see that only one of the terms will be active depending on the value of t. **Logistic regression tries to finds the $\theta$ that maximizes the likelihood $L(\theta)$, which is the same $\theta$ that maximizes the $log-likelihood l(\theta) = log L(\theta)$**.

Before we get started I wanted to familiarize you with some notation.

#### Weighted sum
$$
\theta^{T} \mathbf{x}=\sum_{i=1}^{n} \theta_{i} x_{i}=\theta_{1} x_{1}+\theta_{2} x_{2}+\cdots+\theta_{n} x_{n}
$$


#### Rules for logarithmic expressions

$$
\begin{aligned}
&\log \left(\frac{x}{y}\right)=\log (x)-\log (y) \newline
&\log \left(e^{a}\right)=a
\end{aligned}
$$

#### Rules for derivatives of logarithmic expressions

$$
\begin{aligned}
&\frac{\delta}{\delta x} \log (\text { expression })=\frac{1}{\text { expression }} \cdot \frac{\delta}{\delta x} \text { expression }
\end{aligned}
$$

Let say for instance:
$$
\begin{aligned}
&\frac{\delta}{\delta x} \log (x)=\frac{1}{x} \cdot \frac{\delta}{\delta x} x=\frac{1}{x} \cdot 1=\frac{1}{x}\newline
&\frac{\delta}{\delta x} \log \left(\frac{1}{2 x^{2}+3}\right)=\left(2 x^{2}+3\right) \cdot \frac{\delta}{\delta x} 2 x^{2}+3=\left(2 x^{2}+3\right) \cdot 4 x=8 x^{3}+12 x
\end{aligned}
$$

### Finding the partial derivatives for each j in θ

This is an almost maddeningly clever technique. It is not the sort of answer one might find casually.

#### 1 Simplify the cost function

$J(\theta)=-\frac{1}{m}\left[\sum\limits_{i=1}^{m}\left(\left(\log h_{\theta}\left(x^{(i)}\right)\right)^{y^{(i)}}+ \log \left(1-h_{\theta}\left(x^{(i)}\right)\right)^{\left(1-y^{(i)}\right)}\right)\right]$

**Apply log Power Rule**

$J(\theta)=-\frac{1}{m}\left[\sum\limits_{i=1}^{m}\left(y^{(i)}\left(\log h_{\theta}\left(x^{(i)}\right)\right)+\left(1-y^{(i)}\right) \log \left(1-h_{\theta}\left(x^{(i)}\right)\right)\right)\right]$

 **Replace** $h_{\theta}\left(x^{(i)}\right)$ **with sigmoid**
 
$=-\frac{1}{m}\left[\sum\limits_{i=1}^{m}\left(y^{(i)} \log \left(\frac{1}{1+e^{-\theta^{T} x^{(i)}}}\right)+\left(1-y^{(i)}\right) \log \left(1-\frac{1}{1+e^{-\theta^{T} x^{(i)}}}\right)\right)\right]$
 

**Convert right term to single rational expression**

$=-\frac{1}{m}\left[\sum\limits_{i=1}^{m}\left(y^{(i)} \log \left(\frac{1}{1+e^{-\theta^{T} x^{(i)}}}\right)+\left(1-y^{(i)}\right) \log \left(\frac{e^{-\theta^{T} x^{(i)}}}{1+e^{-\theta^{T} x^{(i)}}}\right)\right)\right]$

 
**Apply Quotient Rule** $\log \left(\frac{a}{b}\right)=\log (a)-\log (b)$ **on left term**

$=-\frac{1}{m}\left[\sum\limits_{i=1}^{m}\left(y^{(i)}\left(\log (1)-\log \left(1+e^{-\theta^{T} x^{(i)}}\right)\right)+\left(1-y^{(i)}\right) \log \left(\frac{e^{-\theta^{T} x^{(i)}}}{1+e^{-\theta^{T} x^{(i)}}}\right)\right)\right]$

$=-\frac{1}{m}\left[\sum\limits_{i=1}^{m}\left(-y^{(i)} \log \left(1+e^{-\theta^{T} x^{(i)}}\right)+\left(1-y^{(i)}\right) \log \left(\frac{e^{-\theta^{T} x^{(i)}}}{1+e^{-\theta^{T} x^{(i)}}}\right)\right)\right]$

 

**Apply Quotient Rule** $\log \left(\frac{a}{b}\right)=\log (a)-\log (b)$ **to right term**

$=-\frac{1}{m}\left[\sum\limits_{i=1}^{m}\left(-y^{(i)} \log \left(1+e^{-\theta^{T} x^{(i)}}\right)+\left(1-y^{(i)}\right) \log \left(e^{-\theta^{T} x^{(i)}}\right)-\left(1-y^{(i)}\right)\left(\log \left(1+e^{-\theta^{T} x^{(i)}}\right)\right)\right]\right.$

 
**Apply** $\log \left(e^{a}\right)=a$ **to right term**

$=-\frac{1}{m}\left[\sum\limits_{i=1}^{m}\left(-y^{(i)} \log \left(1+e^{-\theta^{T} x^{(i)}}\right)+\left(1-y^{(i)}\right)\left(-\theta^{T} x^{(i)}\right)-\left(1-y^{(i)}\right)\left(\log \left(1+e^{-\theta^{T} x^{(i)}}\right)\right)\right]\right.$

**Move minus sign inside** $\sum$

$=\frac{1}{m}\left[\sum\limits_{i=1}^{m}\left(y^{(i)} \log \left(1+e^{-\theta^{T} x^{(i)}}\right)+\left(1-y^{(i)}\right)\left(\theta^{T} x^{(i)}\right)+\left(1-y^{(i)}\right)\left(\log \left(1+e^{-\theta^{T} x^{(i)}}\right)\right)\right]\right.$

**Combine first and third terms**

$=\frac{1}{m}\left[\sum\limits_{i=1}^{m}\left(\log \left(1+e^{-\theta^{T} x^{(i)}}\right)+\left(1-y^{(i)}\right)\left(\theta^{T} x^{(i)}\right)\right)\right]$



#### 2 Take the partial derivative

$\frac{\partial}{\partial \theta_{j}} J(\theta)=\frac{1}{m}\left[\sum\limits_{i=1}^{m}\left(\frac{e^{-\theta^{T} x^{(i)}}\left(-x_{j}^{(i)}\right)}{1+e^{-\theta^{T} x^{(i)}}}+\left(1-y^{(i)}\right) x_{j}^{(i)}\right)\right]$

**Now factor out** $x^{(i)_{j}}$

$=\frac{1}{m}\left[\sum\limits_{i=1}^{m}\left(\frac{-e^{-\theta^{T} x^{(i)}}}{1+e^{-\theta^{T} x^{(i)}}}+1-y^{(i)}\right) x_{j}^{(i)}\right]$

**Combine first two terms**

$\left.=\frac{1}{m}\left[\sum\limits_{i=1}^{m}\left(\frac{1}{1+e^{-\theta^{T} x^{(i)}}}-y^{(i)}\right) x_{j}^{(i)}\right)\right]$

**Substitute** $h_{\theta}\left(x^{(i)}\right)$ for sigmoid function

$$\frac{\partial J(\theta)}{\partial \theta_{j}}= \sum\limits_{i=1}^{m}\left[h_{\theta}\left(x^{(i)}\right) -y^{(i)}\right] x_{j}^{(i)}$$

Such an equation is also known as the **Gradient of Log Likelihood** and can be redifined as follows:

$$
\left[\begin{array}{c}
\frac{\partial J(\theta)}{\partial \theta_{0}} \newline
\frac{\partial J(\theta)}{\partial \theta_{1}} \newline
\vdots \newline
\frac{\partial J(\theta)}{\partial \theta_{n}}
\end{array}\right]=\sum\limits_{i=1}^{m}\left[h_{\theta}\left(x^{(i)}\right) - y^{(i)}\right] x_{j}^{(i)}
$$


