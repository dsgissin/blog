---
layout: post
title: Over-Parameterization and Optimization - A Gentle Start
comments: true
tags: optimization deep-learning
---

Recently, there have been some interesting research directions for studying neural networks by looking at the connection between their parameterization and the function they represent. Understanding the connection between these two representations of neural networks can help us understand why SGD works well in optimizing these networks, and hopefully why SGD and over-parameterization seem to lead to strong generalization...

In this blog series I'll review some of these concepts and develop some interesting ideas for how we can use this view to try and develop new optimization algorithms for neural networks.

{: class="table-of-content"}
* TOC
{:toc}

## Over-Parameterization

Neural networks are over-parameterized, meaning the same function can be represented by different sets of parameters of the same architecture. We can look at a simple example to demonstrate this - a two-layer linear neural network parameterized by the vector \\(v \in \mathbb{R}^{d}\\) and the matrix \\(U \in \mathbb{R}^{d \times d}\\):

$$ f(x)=v^{T}Ux $$

Clearly, this is just a linear function, which means we only need \\(d\\) dimensions in order to define it, but we have more than \\(d^{2}\\). We have many ways of representing the same linear function with different parameters - if we permute the rows of \\(U\\) and the entries of \\(v\\) in the same way, the function will stay the same. If we multiply \\(U\\) by a scalar and divide \\(v\\) by that same scalar, the function will stay the same. These invariances of the model are relevant not only for linear networks, but for piece-wise linear as well (like ReLU networks).

