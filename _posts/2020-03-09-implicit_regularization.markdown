---
layout: post
title: A Primer on Implicit Regularization
comments: true
tags: optimization deep-learning implicit-bias implicit-regularization
---

> The way we parameterize our model strongly affects the gradients and the optimization trajectory. This biases the optimization process towards certain kinds of solutions, which could explain why our deep models generalize so well.

<!--more-->

In the [last blog series][post1], we saw how a deep parameterization can cause the gradients of our model (a quadratic neural network) to be non-convex and a bit weird. That led us to try and develop algorithms for optimizing the network in a linear space, to get all the good theoretical guarantees that classical convex optimization gives us for SGD. We saw that it’s difficult to get these algorithms to work, and even then they’re hard to generalize to large neural networks.

This time, we will lean in to the weirdness. We will analyze how different parameterizations of the same function lead to wildly different optimization dynamics, and how this sort of analysis can help us understand how neural networks generalize so well. Since the last blog series was a bit long, we won’t assume prior knowledge.

All of the plots in this post can be recreated using the jupyter notebook [here][notebook].

{: class="table-of-content"}
* TOC
{:toc}

## Introduction

One of the big questions in theoretical ML today is to try and explain how it is that neural networks generalize so well to unseen examples while being so expressive as a model. For many difficult tasks, neural networks can [memorize random labels][rethinking_generalization], more or less meaning that we shouldn’t expect them to be able to generalize to new examples. There is an infinite number of solutions for these datasets that get a loss of zero (ERM solutions), some of these [generalize very poorly][bad_minima_exist] - if we chose one of these at random, we shouldn’t expect it to generalize.

But we aren’t choosing a solution at random - we use a specific algorithm for finding the solution, and that algorithm biases us towards a specific kind of solution out of all the possible ones. There are types of solutions SGD favors, and characterizing them may help us explain why neural networks generalize so well.

This is the premise of the theoretical study of **implicit regularization** as a way of explaining generalization in neural networks. We want to see how the interplay of gradient based optimization and the parameterization of the function we are optimizing affects the kinds of solutions we get from our model. In this blog post we will use toy models to get a sense of how much of a difference this implicit regularization has on the solution, and get a relatively gentle introduction into this very interesting field.


## The Toyest of Models

While there’s been a lot of work in recent years on implicit regularization in relatively complicated models, we will look at a very simple model - a **linear 2D model**. This will allow us to visualize the gradients of the different parameterizations while still seeing all of the interesting phenomena that exist in larger models.

Our model will be parameterized by $$w \in R^{2}$$ such that for an input $$x \in R^{2}$$, the output will be:

$$ f(x) = \langle w, x \rangle = w_{1}x_{1} + w_{2}x_{2} $$

Very simple. However, the fun will start when we change the parameterization of our model. There will still be a $$w \in R^{2}$$ that is equivalent to our model, but we will be optimizing over slightly different parameters each time.


## The Interplay of Parameterization and Gradient Dynamics

We are going to explore how the gradient dynamics of our model behave in different conditions, so let’s remember how to calculate the gradient of our weights. For a loss function $$\ell(w)$$, where $$w$$ are the linear weights above, the weights move in the negative direction of the gradient (as in gradient descent). Let’s look at an example of a loss function that we will be using[^gaussian_loss]:

$$ \ell(w) = \frac{1}{2}||w-w^{*}||^{2} $$

This loss is simply the squared distance to some optimal set of weights $$w^{*}$$ (which we want to learn). The negative gradient in this case is simple:

$$ \nabla_{w} = w^{*} - w $$

However, things look a little different when we play around with the parameterization. Instead of looking at the gradient of $$w$$, which can be seen as our canonical representation, we will describe $$w$$ with new parameters $$u$$, and optimize over these new parameters.

To make things clear, let’s look at an example. We can define $$w(u)$$ in the following way:

$$ w_{1} = u_{1}^{3} $$

$$ w_{2} = u_{2}^{3} $$

Note that we didn’t really change anything - any function that we could have expressed with $$w$$ can still be expressed with an appropriate $$u$$. While this change seems innocent and harmless (and redundant), it affects the gradients of our model. To compare the parameterizations on the same footing, we would like to look at the gradient of this new parameterization in the canonical representation - we want to see how the gradient of $$w$$ looks, when we optimize over $$u$$. We can use the chain rule for that - we calculate how $$u$$ changes according to the gradient of the loss, and multiply that by how $$w$$ changes with $$u$$. Our general formula:

$$ \nabla_{w(u)} = \nabla_{u} \frac{dw}{du} = \nabla_{w} \frac{dw}{du}^{T}\frac{dw}{du} $$ 

We will use $$\nabla_{w(u)}$$ to denote the gradient of the canonical representation of our function, when it is parametrized by $$u$$. $$\nabla_{u}$$ denotes the gradient of the loss with respect to $$u$$ and $$\nabla_{w}$$ denotes the gradient of the loss with respect to the canonical parameters (without additional parameterization). There is a delicate point here the we should stress - since we are performing gradient descent over $$u$$, the model changes according to $$\nabla_{u}$$ (which we calculate using the chain rule). However, since we want to see how the model changes compared to the canonical parameterization, we need to look at how $$w$$ changes with respect to the change we apply to $$u$$, which gives us the second application of the chain rule. For our specific example, we get:

$$ \nabla_{w(u)_{i}} = 9 (w_{i}^{*} - u_{i}^{3}) u_{i}^{4} = 9 (w_{i}^{*} - w_{i}) w_{i}^{\frac{4}{3}} $$

Writing the gradients only as a function of the canonical parameters $$w$$ shows us that they are surprisingly different from the original gradients - the values of $$w$$ that we optimize are now weighted by the current value of $$w$$. We should expect that values of $$w$$ that are close to $$0$$ will change very slowly, while large values should change rapidly. This is reminiscent of the exploding/vanishing gradient problem of deep neural networks, and it is basically a toy version of it - optimizing over a product of linear parameters causes the gradient’s norm to be strongly dependent on the current values of the parameters.

Now that we have a basic picture of the relevant concepts, we can start playing around with different parameterizations and make some nice visualizations.


### Playing Around With Parameterizations

The nice thing about having a 2D model, is that we can plot the gradient in every point and visualize the gradient field over the loss surface. We can start simple, with the gradients over the canonical parameterization:

$$ \ell(w) = \frac{1}{2} || w - w^{*} ||^{2} $$

$$ \nabla_{w} = w^{*} - w $$

For every $$w$$, we will plot the negative gradient with respect to the loss. Because the size of the gradients will vary a lot in our plots, we will only focus on the direction of the gradients and plot all of them in the same size. The optimal parameters will be shown as a green dot:

{% include image.html path="implicit_regularization/point_identity_param_grad_bw.png" %}

Perhaps unsurprisingly, optimizing the squared loss with gradient descent moves the parameters in the Euclidean shortest path to the optimal solution. This is one of the nice things about optimizing linear models which we explored in previous posts. However, as we’ve said, the gradient field looks quite different for different parameterizations...


### "Deep" Parameterization

The example parameterization we saw before is a special case of a "deep" parameterization - instead of a shallow linear model, we have the parameters raised to the power of the "depth". You can think of this parameterization as a deep neural network without activation functions, where the weight matrices are all diagonal and weight-shared. 

For a depth $$N\ge1$$ model, we parameterize w in the following way:

$$ w_{i}(u) = u_{i}^{N} $$

Calculating the gradients as we did before, we get the following formula:

$$ \nabla_{w(u)_{i}} = N^{2} w_{i}^{2-\frac{2}{N}} (w_{i}^{*} - w_{i}) $$

As before, we can plot the gradient field to see how the parameters move under gradient descent[^gradient_flow]. We will focus on the positive quadrant of the parameter space, to avoid having to deal with even values of N where w can’t be negative under this parameterization:

{% include image.html path="implicit_regularization/2_point_deep_param_grad_bw.png" %}

{% include image.html path="implicit_regularization/4_point_deep_param_grad_bw.png" %}

{% include image.html path="implicit_regularization/6_point_deep_param_grad_bw.png" %}

Immediately we see that these gradient fields are completely different than the previous one - the parameters no longer move in straight lines and they don’t take the shortest path towards the optimal solution. We see that when the values of the two current parameters are different, the gradients become biased towards the direction of the larger value. 

We can make a distinction between the case where the initial values of $$w$$ are smaller than the optimal $$w^{*}$$ and the case where they are larger. When they are smaller, the gradient field has a “rich get richer” dynamic -  large values become larger faster. When they are larger and all values need to become smaller, the larger values become smaller faster and we get a dynamic where the values are biased towards being equal. The deeper the parameterization, the stronger the phenomena.

Assuming we initialize the model with small random values (as is the case in deep learning), this sort of dynamic leads to the model being sparse for a large portion of the optimization trajectory. In the extreme cases, the values are more or less learned one by one.

We see interesting phenomena for deep models, and a reasonable thing to ask is what happens when we keep increasing $$N$$...


### "Infinitely Deep" Parameterization

We could ask how the gradient of the deep parameterization will look like for $$N \rightarrow \infty$$, assuming we normalize the loss by $$N^{2}$$ to not have gradients of infinite size. Surprisingly, this behavior happens when we parameterize our model with an exponent:

$$ w_{i}(u) = e^{u_{i}} $$

$$ \nabla_{w(u)_{i}} = e^{2u_{i}} (w_{i}^{*} - w_{i}) = w_{i}^{2}(w_{i}^{*} - w_{i}) $$

The gradient field in this case is indeed a limiting behavior of the deep parameterization:


{% include image.html path="implicit_regularization/point_exp_param_grad_bw.png" %}


Let’s look at another parameterization - we tried a toy deep parameterization using a larger power than $$1$$, let’s try a fractional power as a toy "anti-deep" model.


### "Anti-Deep" Parameterization

If large positive powers bias us towards sparsity, do fractional powers do the opposite?

For a depth $$0 < N < 1$$ model, we parameterize w in the same way as before, to get us the following gradient fields:

{% include image.html path="implicit_regularization/0.5_point_deep_param_grad_bw.png" %}

{% include image.html path="implicit_regularization/0.2_point_deep_param_grad_bw.png" %}

Interestingly, the behavior of the gradients are indeed opposite to the deep parameterization. Since the gradients are weighted by the inverse of the current weights, larger weights lead to smaller gradients and the optimization process, when initialized with small values, is biased towards having values of similar sizes. If the deep parameterization is of low $$\ell_{1}$$ norm for most of the optimization process, this parameterization has low $$\ell_{\infty}$$ norm.

Finally, we can’t look at 2D parameterizations without exploring one of the classics...

### Polar Parameterization

We can use polar coordinates to parameterize our model in the following way:

$$ w_{1}(r,\theta) = r\cos(\theta) $$

$$ w_{2}(r,\theta) = r\sin(\theta) $$

The gradient field for this parameterization:

{% include image.html path="implicit_regularization/point_polar_point_grad_bw.png" %}

Very cool.

This plot is quite different than the previous ones, and it’s hard to define a clear bias for the path that the model takes until convergence. Note how for small radii the gradients are more inclined to increase the radius, while for radii larger than 1 the gradients have a much stronger angular component. This is caused by the fact that the angular gradient is weighted by the radius, and so we get an angular behavior similar to our deep parameterization.

All of these plots should drive the point home that small changes to the parameterization of the function that don’t affect the expressiveness of our model can lead to completely different optimization behavior. Still, you may ask why we should care so much about the optimization trajectory when all of these models eventually converge to the same solution. Great question.


## Multiple ERM Solutions

While the demonstrations above were for a toy model and a very simple loss function, and so all of the parameterizations converged to the same solution, real models (and neural networks especially) can have many different zero-loss solutions that generalize differently. We should expect parameterizations with different trajectories to converge to different solutions, which means that there would be parameterizations that generalize better than others, for the same class of functions.

To illustrate this, let’s look at a different loss function that has multiple solutions. Instead of the squared distance from an optimal point, we will use a loss that is the squared distance from an optimal line. For an optimal line defined by the following equation:

$$ aw_{1} + bw_{2} + c = 0 $$

Our loss function will be the squared distance from that line:

$$ \ell(w) = \frac{1}{2(a^{2} + b^{2})}  (aw_{1} + bw_{2} + c)^{2} $$

Note that this loss isn't unrelated to real-life models. Up to a scalar multiplication, it is equivalent to the squared loss over a dataset with a single example $$(a,b)$$ with the label $$y=-c$$. Since we have more parameters than examples ($$2$$ vs $$1$$), we can expect there to be multiple optimal solutions (infinite in our case). It is then reasonable to ask which one we converge to and what properties it has.

Doing the same thing we did for the previous loss, we can plot the gradient fields of the different parameterizations (the optimal line is plotted in green):

{% include image.html path="implicit_regularization/line_identity_param_grad_bw.png" %}

{% include image.html path="implicit_regularization/4_line_deep_param_grad_bw.png" %}

{% include image.html path="implicit_regularization/0.25_line_deep_param_grad_bw.png" %}

{% include image.html path="implicit_regularization/line_polar_line_grad_bw.png" %}

We can see that the different parameterizations converge to very different solutions, corresponding to the behaviors we noted for the previous loss. We see that when the function class we’re using to model our data is overly expressive, the parameterization has a very big effect on the solution we end up getting when we use gradient-based optimization. This is especially relevant for real-life nonlinear neural networks, even if describing the exact bias of their parameterization is currently out of our reach.


## Conclusion

Hopefully, the plots and discussion in this post made it clear that gradient-based optimization behaves very strangely under non-canonical parameterizations. In deep nonlinear networks it's hard to even describe the canonical (linear) representation[^nonlinear_canonical], so any gradient-based optimization is bound to have these effects. Assuming we are able to carefully avoid the pitfalls caused by these effects (exploding and vanishing gradients for example), we're left with particular biases towards specific types of solutions.

As we've hinted in this post, it seems that deep parameterizations initialized with small values relative to the solution (current deep learning best practice) bias the trajectory towards sparse solutions, and this could help explain why deep neural networks generalize so well even though they are supposedly too expressive to generalize.

This is an ongoing research direction that is currently restricted to relatively simple models, but the results we have so far suggest that studying the entire trajectory of networks during training is a strong way of understanding generalization.


## Further Reading

If you found this post interesting, you're more than welcome to look into our paper[^my_paper], where we study the deep parameterization described in this post for small initializations and characterize the "incremental learning" dynamics that lead to sparse solutions. We then describe theoretically and empirically how this kind of dynamic leads many kinds of deep linear models towards sparse solutions.

Our research builds on many other works of the past few years that you may also be interested in:

The seminal paper that studied the dynamics of deep linear models was by Saxe et al[^saxe_paper], where the authors derived gradient flow equations for deep linear networks under the squared loss and showed how different depths lead to different analytical solutions for the dynamics of the model. This work has since been extended to gradient descent[^gidel_paper].

Another model that has been studied a lot in this context is matrix sensing, where both the input and model are parameterized as matices instead of vectors, and depth comes from linear matrix multiplication. The sparsity that is induced in these models is of low rank. The first work that seriously studied this model in this context was by Gunasekar et al[^gunasekar_first], followed by many other interesting works[^nadav_paper][^woodworth_paper].

While all of the above works deal with regression, there has also been very interesting work done on exponentially-tailed loss relevant for classification. For linear models over separable data, Soudry et al[^soudry_paper] showed that gradient descent biases the solution towards the max-margin solution (minimum $$\ell_{2}$$ norm solution). However, deeply parameterized models behave differently for classification as well, as shown in Gunasekar et al[^gunasekar_conv]. Especially interesting, convolutional models tend towards sparsity in the frequency domain.

There's also been some initial work on nonlinear models, either empirical[^nakirran_paper] or theoretical with restrictions on the data distribution[^combes_paper][^williams_paper]. 


---
---
<sub></sub>

## Footnotes & References

[^gaussian_loss]: This loss is very simple and useful for these demonstrations, but it is also related to real-life loss functions. If we assume that there is indeed a $$w^{*}$$ that labeled our data, and that our data comes from a standard Gaussian distribution, then the expected squared loss of $$w$$ over the distribution is exactly this loss. This means that for linear models over Gaussian-like data distributions and the squared loss, the loss we'll be discussing is a good approximation of the actual loss over our data.

[^gradient_flow]: Note that the plots we show are only an approximation of the trajectories of gradient descent, since gradient descent uses a non-infinitesimal learning rate. These plots actually describe the trajectories of the optimization under an infinitesimal learning rate, otherwise called "gradient flow". These dynamics are often an excellent approximation of the dynamics under small learning rates, and are a good way of getting a feel for the behavior of the optimization process.

[^nonlinear_canonical]: In the [previous blog series][post1] we examined the canonical space of quadratic neural networks, and in general it is relatively simple to extend that discussion to polynomial networks that only have polynomial activations. However, once we have non-smooth activations that goes straight out of the window and we can't describe the canonical space with a finite number of parametes.

[^saxe_paper]: Andrew M Saxe, James L McClelland, and Surya Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. arXiv preprint arXiv:1312.6120, 2013.

[^gidel_paper]: Gauthier Gidel, Francis Bach, and Simon Lacoste-Julien. Implicit regularization of discrete gradient dynamics in linear neural networks. In Advances in Neural Information Processing Systems, pp. 3196–3206, 2019.

[^gunasekar_first]: Suriya Gunasekar, Blake E Woodworth, Srinadh Bhojanapalli, Behnam Neyshabur, and Nati Srebro. Implicit regularization in matrix factorization. In Advances in Neural Information Processing Systems, pp. 6151–6159, 2017.

[^nadav_paper]: Sanjeev Arora, Nadav Cohen, Wei Hu, and Yuping Luo. Implicit regularization in deep matrix factorization. In Advances in Neural Information Processing Systems 32, pp. 7411–7422. Curran Associates, Inc., 2019.

[^woodworth_paper]: Blake Woodworth, Suriya Gunasekar, Jason Lee, Daniel Soudry, and Nathan Srebro. Kernel and deep regimes in overparametrized models. arXiv preprint arXiv:1906.05827, 2019. 

[^soudry_paper]: Daniel Soudry, Elad Hoffer, Mor Shpigel Nacson, Suriya Gunasekar, and Nathan Srebro. The implicit bias of gradient descent on separable data. The Journal of Machine Learning Research, 19 (1):2822–2878, 2018

[^gunasekar_conv]: Suriya Gunasekar, Jason D Lee, Daniel Soudry, and Nati Srebro. Implicit bias of gradient descent on linear convolutional networks. In Advances in Neural Information Processing Systems, pp. 9461–9471, 2018.

[^nakirran_paper]: Preetum Nakkiran, Gal Kaplun, Dimitris Kalimeris, Tristan Yang, Benjamin L Edelman, Fred Zhang, and Boaz Barak. Sgd on neural networks learns functions of increasing complexity. arXiv preprint arXiv:1905.11604, 2019.

[^combes_paper]: Remi Tachet des Combes, Mohammad Pezeshki, Samira Shabanian, Aaron Courville, and Yoshua Bengio. On the learning dynamics of deep neural networks. arXiv preprint arXiv:1809.06848, 2018.

[^williams_paper]: Francis Williams, Matthew Trager, Claudio Silva, Daniele Panozzo, Denis Zorin, and Joan Bruna. Gradient dynamics of shallow univariate relu networks. arXiv preprint arXiv:1906.07842, 2019.


[post1]: https://dsgissin.github.io/blog/2019/06/16/canonical_spaces_1.html
[notebook]: https://github.com/dsgissin/implicit_regularization
[rethinking_generalization]: https://arxiv.org/abs/1611.03530
[bad_minima_exist]: https://arxiv.org/abs/1906.02613
