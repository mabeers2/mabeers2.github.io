---
layout: page
title: Statistics
permalink: /stats/
---

Below I've provided a few examples of my work in statistics. Most of this work is Bayesian and topics covered range from Generalized Linear Models to time series analysis. 


***

### Quantile Spectral Analysis (Summer 2019)
This is my Master's capstone project, in which I compare and contrast a number of approaches to spectral analysis. The project begins with the implementation of frequentist and Bayesian approaches to producing an estimate of the traditional spectral density. Each of these approaches seeks to describe the time series by identifying the dominant frequency components contributing to the mean of the time series. While useful, traditional spectral analysis has some downsides, such as a lack of robustness to outliers. In addition, traditional spectral analysis is incapable of characterizing periodicity anywhere but the mean of the time series. In recognition of these shortcomings, the project then explores quantile spectral analysis methods that allow one to identify the dominant frequency components at a user specified quantile of the time series. While more computationally demanding, we identify datasets where using quantile approaches yields more information about the time series than traditional approaches. The project was completed under the supervision of Dr. Raquel Prado at UCSC.   
[Capstone Project]({{ site.url }}/assets/final_report_v2.pdf)

**Keywords:** Quantile Spectral Analysis, Time Series Data, R

***

### Hierarchical Extension of a Poisson GLM to Accomodate Overdispersion (Fall 2018)
Given some data on the number of manufacturing faults in fabric rolls of given lengths, we were asked to compare the relative fits of a Bayesian Poisson GLM and a Hierarchical Bayesian Poisson GLM. In this case, there was more variability in the data than the Poisson GLM was able to handle, and the hierarchically extended model performed much better. More details are included in the links below. 

[The assignment]({{ site.url }}/assets/hwk-5.pdf), 
[My solutions]({{ site.url }}/assets/ams274_hw5.pdf), 
[Code and Data](https://github.com/mabeers2/Selected-School-Work/tree/master/Selected%20Homeworks/GLM_HW5)

**Keywords:** Bayesian Hierarchical Poisson GLM, Metropolis-Hastings, Gibbs Sampler, R


***

### Bayesian Multinomial Regression (Fall 2018)

In this problem set, we were asked to complete the following tasks. 
1. Prove that we can model the probability of a binary event taking place using logistic or probit regression. 
2. Fit a Bayesian multinomial regression model to predict the primary food choice of alligators using gender and length as predictors. 
3. Show that, when using continuation-ratio logits, one can decompose a multinomial regression into multiple binomial regressions. Apply this approach to predict the toxicity of a certain chemical to mice, again in a Bayesian way. 

More details are included in the links below. 

[The assignment]({{ site.url }}/assets/hwk-4.pdf), 
[My solutions]({{ site.url }}/assets/hw4_274.pdf), 
[Code and Data](https://github.com/mabeers2/Selected-School-Work/tree/master/Selected%20Homeworks/GLM_HW4)


**Keywords:** Bayesian Multinomial Regression, Metropolis-Hastings, R











