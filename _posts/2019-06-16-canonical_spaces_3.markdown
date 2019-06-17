---
layout: post
title: Over-Parameterization and Optimization - From Quadratic to Deep Polynomials
comments: true
tags: optimization deep-learning polynomial-networks
---

> The promise of projection based optimization in the canonical space leads us on a journey to generalize the shallow model to deep architectures. The journey is only partially successful, but there are some nice views along the way.

<!--more-->

[Last time][post2], we used the canonical representation of neural networks to come up with a new optimization algorithm for shallow quadratic networks, by treating them as a symmetric quadratic form and performing low-rank projections. This algorithm is clearly beneficial theoretically, as it allowed us to easily obtain results that were difficult to develop by analyzing SGD. We also saw a hint of empirical potential, but our projection algorithm can't be practical in reality without adapting it to functions with multiple outputs. If we succeed in generalizing the algorithm to multiple outputs, it won't only allow us to optimize shallow multi-class classifiers, but will also allow us to embed the quadratic model as a "quadratic layer" within deep neural networks, while optimizing with projections instead of with regular SGD.

One would think that adding additional outputs shouldn't be a big deal, but it's not as straightforward as one would hope and requires some changes to the algorithm and model...

{: class="table-of-content"}
* TOC
{:toc}

## A New Quadratic Model

When we had a single output, we could look at out model as a quadratic function that is canonically parameterized as a symmetric matrix $$A \in \mathbb{R}^{d \times d}$$. The projection step from a general $$A$$ to a neural network with $$r$$ hidden neurons was then just an eigendecomposition of $$A$$, where the assumption was that the rows of $$W$$ could be taken to be orthogonal without hurting the expressivity of the neural network[^orthogonality]. While this is true for a single output, the orthogonality of $$W$$ becomes restrictive when we have several outputs.

Having $$n$$ outputs for the same quadratic layer basically means that the projection step requires finding a shared $$W$$ for $$\{A_{n}\}_{i=1}^{n}$$, with every $$A_{i}$$ being distinct only through it's $$\alpha_{i}$$ vector. Since the $$W$$ are shared and we aren't guaranteed that the different $$A_{i}$$s share the same spectrum, constraining the $$W$$s to be orthogonal is restrictive and the natural projection using an eigendecomposition of the mean of the $$A_{i}$$s becomes sub-optimal. In other words, the following optimization problem can't be solved analytically:

$$ \underset{W,\alpha}{argmin} \sum_{i=1}^{n} || A_{i} - \sum_{r}\alpha_{i,r}w_{r}w_{r}^{T} ||_{F}^{2} $$

### Adapting the Model

Clearly, the projection algorithm for the quadratic model can't be useful in a practical sense without it being relevant for multiple output functions. This leads us to try to change the model in order to allow for projections. The first step towards having a model that allows projections, is to observe that the parameterization of our model can be rewritten in the following way:

$$ A = \sum_{i=1}^{r} \alpha_{i}w_{i}w_{i}^{T} = W^{T}DW $$

$$ D_{i,i} = \alpha_{i} $$

In words, we can look at $$A$$ not only as a sum of rank-$$1$$ matrices, but also as it's matrix eigendecomposition. In this view, we know that we can't project several matrices onto the same $$W$$ with different diagonal $$D$$ matrices, but maybe if we relax the parameterization and **allow $$D$$ to be symmetric**, we could get away with the projection...

Renaming $$D$$ as $$S$$, our parameterized model will now consist of $$n$$ matrices with the following parameterization:

$$ A_{i} = W^{T}S_{i}W $$

Where $$S_{i}$$ is symmetric and $$W \in \mathbb{R}^{r \times d}$$. We will soon see that this extension of our model allows for a new projection-based optimization algorithm, but how does this new model compare to the original parameterization? Is it even still a neural network?

### Analyzing the New Model vs the Old

Relaxing the model to allow for a symmetric matrix instead of a diagonal one changes the model, so we should first understand the differences between the two models.

#### Computation

The first thing we should understand with this model, is how it is computed. In the original parameterization, the fact that $$D$$ was diagonal allowed us to look at every row of $$W$$ in parallel and use the squared activation function instead of explicitly calculating $$w_{i}^{T}x$$ twice. This made the quadratic function look like a neural network, having a fully connected layer followed by an element-wise non-linear activation. 

This efficient representation and calculation sadly goes away when we allow a symmetric $$S$$, since there is interaction between the rows of $$W$$. The function we are now trying to compute is the following:

$$ A_{i} = x^{T}W^{T}S_{i}Wx $$

We can see that there is still computation that can be shared between the different matrices - all matrices have a quadratic form applied to $$Wx$$. This means we can calculate the $$n$$ functions efficiently by first applying a fully connected layer to $$x$$, moving from $$d$$ dimensions to $$r$$ ($$h=Wx$$). Then, instead of an element-wise activation, we run a quadratic form on the hidden layer for every output ($$h^{T}S_{i}h$$).

So, our model is a bit less "neural networky" since it has vector activations instead of element-wise activations, but it still has the property of sequential computation and efficiently representing $$n$$ functions with parameter sharing. This model is a bit less easy to draw in a network diagram, but here goes nothing:

{% include image.html path="canonical_spaces_3/diagram.png" %}

#### Parameter Count & Computational Complexity

Allowing $$S$$ to be symmetric instead of diagonal introduces more parameters to our model, so we should make a comparison between the two models to try and see when the new model is reasonable.

The original single-output model has $$W \in \mathbb{R}^{r \times d}$$ and $$\alpha \in \mathbb{R}^{r}$$, which means that overall we had $$r(d+1)$$ parameters. Assuming we have $$n$$ outputs and train the old model with regular SGD, $$\alpha$$ becomes a matrix and the number of parameters becomes $$ N_{old} = r(d+n) $$. The new model has the same $$W$$ matrix, but this time every $$S$$ requires $$ r(r+1) $$ parameters. This means that the number of parameters for the new model is $$ N_{new} = dr + nr(r+1) $$

If we compare the two models asymptotically in parameter count, we get that $$ N_{old} = O(rd + rn) $$ and $$ N_{new} = O(rd + r^{2}n) $$. This means that the two models are comparable in parameter count when $$ r \approx \sqrt{d} $$.

The computational complexity of a forward pass between the two models coincides with the number of parameters asymptotically, and the analysis is similar...

#### Expressivity

The final comparison we'll run before moving to optimizing our new model, is an expressivity comparison. Namely, which model is more expressive for every $$r$$.

Clearly, for a given $$r$$ the new model is more expressive than the old model, since $$S$$ being symmetric still allows it to be diagonal. This means that if we define $$ \mathcal{H}_{diag}^{d,r,n} $$ as the hypothesis class of the old model with $$r$$ hidden neurons and $$n$$ outputs and $$ \mathcal{H}_{sym}^{d,r,n} $$ as the hypothesis class of the new model with $$r$$ hidden neurons and $$n$$ outputs, we have the containment:

$$ \mathcal{H}_{diag}^{d,r,n} \subset \mathcal{H}_{sym}^{d,r,n} $$

However, can we bound the new model's expressivity using the old model with more hidden neurons and get a two-way containment?

The answer is yes!

We can build a basis for all symmetric matrices in $$\mathbb{R}^{r \times r}$$, made up of rank-1 matrices. Since the space of symmetric matrices is $$r(r+1)$$ dimensional, we can build such a basis using $$r(r+1)$$ matrices. One possibility for such a basis, is the following - $$ \forall j \le i: (e_{i}+e_{j})(e_{i}+e_{j})^{T} $$.

Now that we have such a basis, we can construct an old quadratic model with $$r(r+1)$$ hidden neurons in the following way: 

We start with the original $$W$$ from a given new model and get an output of $$r$$ hidden neurons. Now, we add another matrix multiplication $$U$$ that has $$r(r+1)$$ outputs, each one being a basis vector. The two matrices can be multiplied to form the model's actual $$W$$ matrix, which is followed by the squared activation. This construction causes the hidden layer's activations to be the projection of $$(Wx)(Wx)^{T}$$ on every basis matrix. Since the basis spans the space of symmetric matrices in $$\mathbb{R}^{r \times r}$$, every symmetric matrix can be expressed as a linear combination of these activations and so every $$S_{i}$$ can be encoded using an $$\alpha$$ vector over the hidden outputs.

This construction is the equivalent of taking the input and raising it explicitly to the degree-2 polynomial linear space, which allows for a linear function to represent any degree-2 polynomial in that space. This construction finally gives us the two-way containment we wanted, namely:

$$ \mathcal{H}_{diag}^{d,r,n} \subset \mathcal{H}_{sym}^{d,r,n} \subset \mathcal{H}_{diag}^{d,r(r+1),n} $$

We see that while the two models are different in many ways, they are still simply two ways of describing degree-2 polynomials, and so they aren't so different. This also gives us an understanding of the difference in parameter count between the two models - if we want the old model to be as expressive as the new model, we need to make it have a similar amount of parameters overall (have $$r(r+1)$$ hidden neurons, leading to similar parameter count).



## The GLRAM Projection Algorithm

Now that we've defined our new model and convinced ourselves that it isn't that different from the old one in terms of the expressivity and parameter count tradeoff, we can move on to deriving a projection algorithm. For that, we will look at a generalization of low-rank approximation derived originally for data in matrix form, called [GLRAM][GLRAM] (Generalized Low-Rank Approximation of Matrices).

### GLRAM's Original Objective - Dimensionality Reduction

### The Algorithm



## Deep Polynomial Networks

TODO: explain how stacking these layers leads to higher degree polynomials - the degree is exponential in the depth!

### Combining Backpropagation and GLRAM Projections

TODO: explain how this is a play between the canonical representation and the deep one, treating every layer in it's canonical form but not expressing the actual canonical form of the entire model...

### Experiments & Issues Scaling Up

TODO: run multiclass examples without the linear layer and see if we improve... Do the same for convolutional layers and CIFAR

TODO: explain in summary that we got good algorithms, but in general it seems that working in the canonical space isn't something that can scale up to deep networks... So maybe what we should do is try to make the dynamics in parameteric space more canonical (in the next post).

---
---
<sub></sub>

## Footnotes

[^orthogonality]: We needed the orthogonality assumption in order to analytically solve the projection optimization objective using an eigendecomposition. For a single symmetric $$A$$, this orthogonality assumption wasn't restrictive since every symmetric matrix can be decomposed into orthogonal eigenvectors.

[post2]: https://dsgissin.github.io/blog/2019/06/16/canonical_spaces_2.html
[GLRAM]: https://bit.csc.lsu.edu/~jianhua/neelavardhan.pdf
