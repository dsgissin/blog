---
layout: post
title: Over-Parameterization and Optimization - A Gentle Start
comments: true
tags: optimization deep-learning
---

> Excerpt Here

<!--more-->

Recently, there have been some interesting research directions for studying neural networks by looking at the connection between their parameterization and the function they represent. Understanding the connection between these two representations of neural networks can help us understand why SGD works well in optimizing these networks, and hopefully why SGD and over-parameterization seem to lead to strong generalization...

In this blog series I'll review some of these concepts and develop some interesting ideas for how we can use this view to try and develop new optimization algorithms for neural networks.

{: class="table-of-content"}
* TOC
{:toc}

## Over-Parameterization

Neural networks are over-parameterized, meaning the same function can be represented by different sets of parameters of the same architecture. We can look at a simple example to demonstrate this - a two-layer linear neural network parameterized by the vector $$v$$ and the matrix $$U \in \mathbb{R}^{d \times d}$$:

$$ f(x)=v^{T}Ux $$

Clearly, this is just a linear function, which means we only need $$d$$ dimensions in order to define it, but we have more than $$d^{2}$$. We have many ways of representing the same linear function with different parameters - if we permute the rows of $$U$$ and the entries of $$v$$ in the same way, the function will stay the same. If we multiply $$U$$ by a scalar and divide $$v$$ by that same scalar, the function will stay the same. These invariances of the model are relevant not only for linear networks, but for piece-wise linear as well (like ReLU networks).

You may ask - so what? As long as we are able to express a rich and relevant family of functions with our parameterized model, why should this over-parameterization matter to us?

We will now see that this over-parameterization leads to a very different optimization landscape for SGD.

### Parameteric and Canonical Spaces

If we want to compare the two-layer linear network with the family of linear functions, we will need to define the two parameterizations and the way they are connected to each other. We already know what the parameteric space is - this is the over-parameterized model defined by $$U$$ and $$v$$. The canonical representation in our case will be the space of linear functions in $$\mathbb{R}^{d} \rightarrow \mathbb{R}$$, parameterized by $$w \in \mathbb{R}^{d}$$.

We call the canonical space "canonical", because every set of parameters in it define a unique function, and every set of parameters in the parameteric space has a corresponding function in the canonical space. We are dealing with linear functions for now, so mapping from the parameteric space to the canonical is simply a matrix multiplication:

$$w(U,v)=U^{T}v$$

Note that we can't map back from the canonical space to the parameteric, because the mapping isn't bijective - a single $$w$$ has many parameterizations in the parameteric space... Also, note that while we are dealing with linear functions, the mapping between the two spaces is not linear in $${U,v}$$.

### Gradient Dynamics Between Spaces

Now that we've established these two spaces, both essentially representing the same functions, we can compare their gradient dynamics. Given an example $$x \in \mathbb{R}^{d}$$ and our convex loss function $$\ell(f(x),y)$$, we can write down the gradient of the loss with respect to the general parameters \theta:

$$\frac{\partial \ell}{\partial \theta}=\frac{\partial \ell}{\partial f}\frac{\partial f}{\partial \theta}$$

We see that the parameters are only present in the derivative of the function $$f$$, so we can ignore the gradient of the loss and focus the two parameterizations. For the canonical space, the gradient is just the gradient of a linear function:

$$f(x) = w^{T}x$$

$$\frac{\partial f(x)}{\partial w} = x$$

For the parametric representation, we can use the chain rule to get the gradient (or calculate it directly):

$$f(x) = v^{T}Ux$$

$$\frac{\partial f(x)}{\partial v} = \frac{\partial f(x)}{\partial w}\frac{\partial w(U,v)}{\partial v} = Ux$$

$$\frac{\partial f(x)}{\partial U} = \frac{\partial f(x)}{\partial w}\frac{\partial w(U,v)}{\partial U} = vx^{T}$$

Now we start to see something interesting in the dynamics of the deep representation - while in the canonical space all functions have the same gradient with respect to $$w$$ (since the space is linear), **in the deep representation the gradient depends on the parameterization**. For example, given a fixed parameterization, if we increase the norm of all of the weights - the gradient will have a larger norm as well. This means that as training progresses and the norm of our weights increase (as they generally do), the gradients tend to grow as well (ignoring the gradient of the loss which usually becomes smaller).

We can see even more interesting phenomena when we look at the models after a small gradient step. Assuming we use an infinitesimal learning rate $$\eta$$, the model after a canonical gradient step will be:

$$\hat{f} = w - \eta \ell^{'}x$$

However, the model after a parameteric gradient step will be (ignoring second order terms of $$\eta$$):

$$\hat{f} = (v - \eta v^{'})^{T}(U - \eta U^{'}) = U^{T}v - \eta (U^{T}Ux + v^{T}vx) = U^{T}v - \eta (U^{T}Ux + ||v||^{2}x) $$



