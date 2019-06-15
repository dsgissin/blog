---
layout: post
title: Over-Parameterization and Optimization - A Gentle Start
comments: true
tags: optimization deep-learning
---

> Deep, over-parameterized networks have a crazy loss landscape and it's hard to say why SGD works so well on them. Looking at a canonical parameter space may help.

<!--more-->

Recently, there have been some interesting research directions for studying neural networks by looking at the connection between their parameterization and the function they represent. Understanding the connection between these two representations of neural networks can help us understand why SGD works well in optimizing these networks, and hopefully why SGD and over-parameterization seem to lead to strong generalization...

In this blog series I'll review some of these concepts using toy examples, and develop some interesting ideas for how we can use this view to try and develop new optimization algorithms for neural networks.

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

$$\hat{f} = (v - \eta \ell^{'} v^{'})^{T}(U - \eta \ell^{'} U^{'}) = U^{T}v - \eta \ell^{'} (U^{T}Ux + v^{T}vx) = U^{T}v - \eta \ell^{'} (U^{T}Ux + ||v||^{2}x) $$

And look at this - we already know that the deep parameterization affects the norm of the gradient, but now we see that **the deep parameterization also affects the direction of the gradient**. The direction of the canonical gradient is along the axis defined by the input $$x$$, but the parameteric gradient is along the axis of the vector $$U^{T}Ux +  \left\lVert v\right\rVert^{2}x$$.

### A Variety of Parametric Gradients

Taking the example above, it is interesting to explore how varied the direction and norm of the gradient can be, depending on the parameterization of the same function. There is a general transformation to the parameters of the deep model that we can look at that make the function remain the same - if we multiply $$v$$ by some invertible matrix $$P$$, and multiply $$U$$ by it's inverse, we are left with the same function we started with:

$$(P^{-T}v)^{T}PU = v^{T}P^{-1}PU = v^{T}U$$

Now, this means that while a given function has a canonical gradient with a unique direction and norm, that same function in a two-layer parameterization can have many possible gradient directions and norms - any defined by an invertible $$P$$:

$$(U^{T}P^{T}PU + ||P^{-T}v||^{2}I)x$$

Basically, by choosing the right $$P$$ we can get a gradient to point anywhere we want in the positive halfspace of $$x$$ (since the matrix multiplying $$x$$ is PSD)![^halfspace]

This is very interesting. The canonical space of functions we are working with is linear and we know a lot about how optimization works in it (namely, we can easily prove convergence for convex losses). However, once we over-parameterize a little we get completely different optimization dynamics which are much harder to deal with theoretically. These over-parameterized gradient dynamics depend on the parameterization at every time step, which means that if we want to guarantee convergence **we have to pay attention to the entire trajectory of the optimization**...

### Take-Aways From the Linear Example

This 2-layer linear example is far from a deep convolutional ReLU network, but we can already see interesting phenomena in this example that exist in optimization of real networks.

The main thing we can gather from this example is the **importance of balance netween the norms of the layers** at initialization and during the optimization process. Looking at the parameteric gradient, if the norm of $$v$$ is much larger than the norm of $$U$$, then the gradient of the function will be huge (and the same goes for large $$U$$ and small $$v$$). This means we need to initialize the parameters such that they have similar norms, like we do in practice. This generalizes to real neural networks as the exploding/vanishing gradient problem...

Another interesting thing we can gather, is the $$possible importance of learning rate decay for deep models$$. For example, if the minimum of our loss function was some linear function with a very large $$\ell_{2}$$ norm, than a large learning rate could cause the dynamics to blow up when the parameters become large (since the gradient norm grows with the parameters). However, a small learning rate would take a very long time to converge. This suggests a learning rate schedule that is correlated with the norm of our model (more on that later).

## A More General View

In the next post I'll be exploring non-linear neural networks, but before we get there we should take a step back and try to generalize what we just saw for deep linear networks.

We had two ways of looking at the same family of functions. The first was a canonical parameterization, where every function in our family had a unique parameterization and the function space was linear. The second was an over-parameterized representation, where every function in our family had many possible parameterizations. We also had a non-linear mapping from the parameteric space to the canonical one. Let's define the canonical parameters as $$\Theta=\mathbb{R}^{q}$$ and the deep parmeterization's parameters as $$\mathcal{W}=\mathbb{R}^{p}$$. The mapping between some $$W$$ and some $$\theta$$ can be written as $$\Psi:\mathcal{W} \rightarrow \Theta$$.

So far we looked at how the gradients behave in the two spaces for the linear 2-layer example, but it should be interesting to see more generally how the loss landscape looks like between the two spaces.

### The Loss Landscape

The first thing we care about in a loss landscape, is where are the critical points and are they well behaved?

For the canonical representation, if we assume that the functions in this representation are linear (like in our example and in future examples), then if the loss is convex we are very happy - there is a unique minimum to our function and we are guaranteed to reach it using SGD![^kernel]

As for the parametric representation, things aren't necessarily as simple. As a quick example, we can look back at out linear example where $$U=0$$ and $$v=0$$. Looking at the gradients, this is a critial point of the loss landscape no matter what loss we have, and it generally isn't a minimum (it's a saddle point when the optimal function isn't the zero function). So, if the canonical space has a unique critical point (the global minimum) but the parameteric space has more than one critial point - where did the additional critical points come from?

Well, we know how the two parameterizations are connected - they are connected by $$\Psi$$. So, we can look like before at the gradients in the two spaces and ask when they are zero:

$$\frac{\partial f}{\partial W} = \frac{\partial \Psi(W)}{\partial W}\frac{\partial f}{\partial \Psi(W)} = \frac{\partial \theta}{\partial W}\frac{\partial f}{\partial \theta}$$

For a given parameterization $$W$$, the canonical and parameteric gradients are connected by a linear transformation define by the matrix $$\frac{\partial \theta}{\partial W} \in \mathbb{R}^{p \times q}$$. We immediately see that if $$\Psi(W)$$ maps to the unique global minimum of the canonical space, then we are at also at a critical point (a global minimum) of the parameteric space, since the $$\frac{\partial f}{\partial W} = 0$$. This is a nice sanity check...

However, the additional critical points come up in the situations where $$\frac{\partial f}{\partial W} = 0$$ while $$\frac{\partial f}{\partial \Psi(W)} \ne 0$$. This can happen for $$W$$s where **the linear transformation between the two gradients is not full rank**. In such a case, $$\frac{\partial \theta}{\partial W}$$ has a non-empty kernel and non-zero canonical gradients can be mapped to zero parameteric gradients, which means that we get a critical point where there is no such critical point in the canonical space.

### The Formation of "Ghost" Saddle Points

So, For which $$W$$ is $$\frac{\partial \theta}{\partial W}$$ of partial rank?

To understand this, we need to more explicitly define $$\frac{\partial \theta}{\partial W}$$. Every column of this matrix defines how a single entry of $$W$$ changes every entry of $$\theta$$. If a set of these columns is linearly dependent, or equal to zero, the matrix can be of partial rank.

This happens for example when two row vectors of the same weight matrix in a neural network are identical/parallel (meaning two neurons are identical) - in such a case the two sets of columns will be linearly dependent. Another example can be "dead neurons" in ReLU networks - these neurons are always zero and so the outgoing weights from them don't effect the actual model (and so their column is a zero vector).

In practice, when the neural network is large enough and the initialization is good, we don't see this happening and the network is able to converge to a global minimum (with a loss of zero). Hui Jiang build on the above kind of reasoning in his [paper][Jiang] to explain why highly expressive neural networks don't get stuck in local minima even though there are so many in the loss landscape.

### The Exploding and Vanishing Gradients Problem

Just like we explored the loss landscape using $$\frac{\partial \theta}{\partial W}$$, we can do the same sort of analysis to explain more generally why there are exploding and vanishing gradients.

Since $$\frac{\partial \theta}{\partial W}$$ depends on $$W$$, it is reasonable to believe (and it is the case in practice) that there are many $$W$$s for which the operator norm of $$\frac{\partial \theta}{\partial W}$$ is very large or very small. In such cases even though the canonical gradient is of a reasonable norm (assuming $$x$$ has bounded norm), we could have $$\frac{\partial \theta}{\partial W}$$ increase/decrease the norm of the gradient considerably, causing the gradient to vanish or explode.

## Further Reading

This sort of comparison between deep and canonical representations is used to both understand why neural networks are able to reach global minima of the loss landscape, and recently to start showing why they generalize well. In the next posts we'll try exploring how we can develop optimization algorithms using this view, that hopefully perform better than SGD on the deep representation.

### Deep Linear Networks

A few papers from Sanjeev Arora and Nadav Cohen, along with other collaborators, address the dynamics of optimizing deep linear networks (deeper than our linear example). 

The [first paper][Nadav1] studies depth and it's effect on optimization. It shows that under certain assumptions on the weight initialization, depth acts as a preconditioning matrix at every gradient step (similar to the PSD matrix we saw in our small example). They also show that this kind of preconditioning cannot be attained by regularizing some type of norm of the canonical model - over-parameterization it is a different kind of animal.

In the [second paper][Nadav2], the authors extend their results and show a convergence proof for deep linear networks under reasonable conditions on the random initialization.

In their recent, [third paper][Nadav3], they move to studying generalization by showing that depth biases the optimization towards low rank solutions for a matrix completion/sensing task. There have been previous results showing that SGD creates this sort of bias and it is a strong belief today that SGD is a main factor in the generalization of neural networks. This work shows that not only does SGD bias us towards simple solutions, but that over-parameterization may also be a factor. As in the first paper, their results suggest that depth is a different animal than regularizing a norm (nuclear or otherwise), being more biased towards low rank than norm regularizations.

### Non-Linear Networks

This sort of analysis is very nice for linear networks where we can clearly define the canonical representation, which happens to be linear and behaves nicely. However, when we move to deep ReLU networks for example, we don't even know how to properly describe the canonical representation, and it is incredibly high dimensional. Still, there are a couple of works that try to use the connection between the two spaces to explain why SGD works in the deep representation.

In Hui Jiang's [paper][Jiang], the analysis of $$\frac{\partial \theta}{\partial W}$$ (refered to as the "disparity matrix") is used to explore the loss landscape of general neural networks, assuming they are expressive enough. The canonical representation that is used is the Fourier representation of functions over the input space (which is also linear and nicely behaved). Assuming the family of neural networks $$\epsilon$$-convers that Fourier space (a strong assumption), this suggests that under a random initialization it would be very hard to have the diparity matrix not be full rank, and therefore we shouldn't be surprised that optimizing with SGD finds the global minimum.

Another [paper][Julius] by Julius Berner et al, analyzes shallow ReLU networks in order to start showing the connection between the parameteric space and canonical space (referred to as "realization space") for an actual neural network. The main result shows "inverse stability" of the shallow neural network under certain, mild conditions on the weights. Informally, inverse stability is the property such that for a given parameterization $$W$$ and it's corresponding canonical representation $$\theta$$, all close canonical representations to $$\theta$$ have a corresponding parametric representation near $$W$$. Such a property suggests that optimizing in the parameteric space should behave like optimization in the canonical space. Another interesting thing in this paper, is the fact that there is explicit discussion of the fact that while the canonical space is linear (as we saw throughout this blog), shallow ReLU networks aren't expressive enough to fill that entire space. This means that the optimization objective in the canonical space is a convex loss function with a non-convex feasible set of solutions. We'll see another, more digestible example of this in the next blog post.

[^halfspace]: The restriction to the halfspace of $$x$$ is true when $$\eta$$ is infinitesimal, otherwise we can't neglect the $$\eta^{2}$$ term. In such a situation, which is what we see in actual SGD, the learning rate also plays a role in determining the direction of the gradient step. Also, large learning rates could even cause the gradient to step out of the halfspace of the canonical gradient, leading to a gradient step negatively correlated to the canonical one.
[^kernel]: Note that a linear canonical space is relevant for any kernel function simply by looking at the reproducing Hilbert space of the kernel. This means that we can mostly be safe in saying that there is some canonical representation which is linear. This space may be infinite-dimensional, but let's not worry about that too much for now...

[Jiang]: https://arxiv.org/pdf/1903.02140.pds
[Nadav1]: https://arxiv.org/pdf/1802.06509.pdf
[Nadav2]: https://arxiv.org/pdf/1810.02281.pdf
[Nadav3]: https://arxiv.org/pdf/1905.13655.pdf
[Julius]: https://arxiv.org/pdf/1905.09803.pdf