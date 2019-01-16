---
layout: page
title: Machine Learning
permalink: /ML/
---

Below I've provided a few examples of my work in Machine Learning.

***

### Multilayer Perceptron Classifier Implementation
In this project, I derive the correct equations for fitting a multilayer perceptron classifier. I compare the results of this implementation to the scikit learn implementations of a MLP classifier and a SVM. My implementation achieves an 91% classification accuracy on the Iris dataset, while the scikit learn MLP classifier implementation achieves 96% percent classification accuracy. Accuracy on my approach could be improved by using a better solver, rather than basic gradient descent or possibly by adding a bias term to each layer of the neural network. These changes may be made at a later date. 

[The results]({{ site.url }}/assets/MLPIRIS.html)

**Keywords** MLP Classifier, Neural Network, Gradient Descent, Backpropogation, Python, Support Vector Machine, Jupyter Notebook

***

### Linear Regression & Naive Bayes Text Classification
In this problem set, we were asked to:

1. Simulate some numerical data with noise and fit a few linear regressions, with and without ridge regularization to this data to find best fitting coefficients. While regression coefficients have a closed form expression in both cases, this problem was made more challenging by the fact that some of the matrices we were trying to invert were near singular so a numerical minimization approach had to be implemented. 

2. Implement a Naive Bayes Classifier to classify emails as ham or spam. We also identified the words that discriminated between ham and spam most strongly. 

More details are included in the links below. 

[The assignment]({{ site.url }}/assets/ml-hw3.pdf), 
[My solutions]({{ site.url }}/assets/cs242_hw3.pdf), 
[Code and Data](https://github.com/mabeers2/Selected-School-Work/tree/master/Selected%20Homeworks/ML_HW3)


**Keywords:** Linear Regression, Optimization, Naive Bayes Text Classification, Python, R




