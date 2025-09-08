---
layout: default
title: Machine Learning
permalink: /machine_learning
markdown: kramdown
---

<!-- *Under construction* -->

#### CONTENTS
- [An old problem with an old solution](#an-old-problem-with-an-old-solution)

<hr>

# An old problem with an old solution
["Runge's phenomenon"](https://en.wikipedia.org/wiki/Runge's_phenomenon) is an observation, published by Carl Runge in 1901, related to the interpolation of a function by polynomials using equidistant nodes. Today, the observation is somewhat trivial or, at least, mainstream; however, it drew attention when it was first made because the standards or rigour among physicists were quite lax back then. Even mathematics had only recently started getting serious with the help of Cauchy and Weierstrass. But, if Dieudonné has written about a topic, there is no point in trying to do better. And he *has* made a remark about Runge's phenomenon (Calcul Infinitésimal, p. 16):

![Dieudonné, Calcul Infinitésimal](/assets/snips/runges_phenomenon/dieudonne_runge_phenomenon.png)

Loosely translated.

<!-- 
But where physicists venture into dangerous territory is when they tend to accept as "obvious" what is no longer so at all, and forget that our "intuition" is a rather rudimentary tool that grossly deceives us on occasion. Contrary to what many of them believe, it is not necessary, to challenge results they accept without discussion, to seek out functions as "monstrous" as functions without derivatives; the "phenomenon of Runge" (chap. IX, Appendix) shows that the classical polynomial interpolation procedure can very well diverge for analytic functions as "excellent" as one could desire; and there are analytic functions for $$|z|\lt 1$$, continuous throughout the disk $$|z|\leq 1$$, and yet which transform the circle $$|z| = 1$$ into a Peano curve filling a square.
 -->

Because there are many moving pieces in what follows, it is useful to have from the beginning a *conceptual* summary of what Runge's phenomenon is about: it simply says that the *obvious* choice of interpolating polynomials can be a terrible idea. It does *not* say that polynomials are useless. It's not a hard problem, let alone an unsolved one. Learning about it, however, can be informative.

## Runge's function
First, let's define Runge's function, which is the function he used as an example in his [paper](https://archive.org/details/zeitschriftfrma12runggoog/page/224/mode/2up?view=theater) on interpolation at equidistant points. But, to make things a bit more dramatic, let's add a coefficient of $$10$$ in the denominator:

![Runge's example function](/assets/snips/runges_phenomenon/runges_function_original.png)

```python
def ground_truth(x: np.array, noise: bool = False, seed: int = None) -> pd.DataFrame:

    if noise:
        if seed is None:
            raise ValueError('If `noise is True` then `seed` has to be of type `int`.')
        rng = np.random.default_rng(seed=seed)
        noise_vector = rng.normal(loc=0.0, scale=0.01, size=x.shape)
    else:
        noise_vector = np.zeros(shape=x.shape)

    return (
        pd
        .DataFrame({
            'x': x,
            'y': np.vectorize(lambda t: 1 / (1 + 10 * t ** 2))(x) + noise_vector
        })
        .sort_values(by=['x'])
        .reset_index(drop=True)
    )
```

Since we're playing around with interpolation, it makes sense to have the option to add noise on a whim. Hence the extra lines of code.

## The problem

If you're all wide-eyed when it comes to approximation theory, the following may seem counter-intuitive.

Suppose that 

$$x_k, \quad k=0,\ldots,n-1, \quad x_k\in[0,1]$$

are equidistant nodes in $$[0,1]$$ and $$y_k\triangleq f(x_k)$$ are the corresponding values of the Runge function. Then, the unique polynomial $$p_{n-1}^L$$ of degree $$n-1$$ that passes through the points $$(x_k,y_k)$$ becomes a terrible approximation (this is a technical term) of the function $$f$$ as $$n$$ increases.

![Lagrange interpolating polynomial for the Runge function](/assets/snips/runges_phenomenon/lagrange_approx_runge.png)

## A classical solution to the problem

There is a classical well-known remedy to this problem: choose your nodes wisely. Specifically, since all truths in the real line pass through the complex plane (according to Hadamard), choose the nodes to be projections of equidistant points (the distance being measured by arc length) on the unit circle. In other words, a better choice is the *Chebyshev points of the 1st kind*. 

![Chebyshev interpolating polynomial for the Runge function](/assets/snips/runges_phenomenon/chebyshev_approx_runge.png)

Because eyeballing pixels when it comes to approximation errors can be extremely useful but also a terrible idea (and you don't know which applies when), here are also the errors of the Chebyshev interpolation:

![Chebyshev interpolation errors](/assets/snips/runges_phenomenon/cheb_interp_errors.png)

That the graphs of the absolute and relative errors look identical is a coincidence due to the range of values of $$x$$ and $$y$$. No need to worry about that. There is no sleight of hand here; Chebyshev polynomials do work extremely well for certain tasks, and it's an entire rabbit hole to go down if one is willing to. An excellent book of applied mathematics that demonstrates some powerful techniques, especially for root finding, is [J. Boyd's book](https://www.google.ca/books/edition/Solving_Transcendental_Equations/29CgBAAAQBAJ?hl=en&gbpv=0). There is also the whole "chebfun" business that spun out of these ideas &mdash; google is your friend. The overarching idea is that "continuous mathematical objects" (whatever that means) can be replaced or proxied up to machine precision by their Chebyshev approximations. But there is more magic to it, actually: from the proxy one can compute roots, derivatives, and integrals. The major limitation of this paradigm is the inability to handle high-dimensional problems, but 'nuff said for now (and, no, "tensor products" have not fully addressed the limitation).

## A different problem

At this point we know that:

1. Exact interpolation at equidistant points *can* be a terrible idea.
2. Exact interpolation at cleverly sampled points *can* be a fantastic idea.

It all depends on the task at hand. But one can ask **other** interesting questions. For example, is a good interpolant, such as the one provided by the Chebyshev interpolation, somehow inherently tied to the function that generated the interpolated values? Or, put differently, is a good interpolant "learning" the data generation mechanism? Now, a mathematically literate person (say, someone who has been exposed to post 18-th century mathematics) should immediately offer an emphatic "no" or an emphatic "no, unless you prove otherwise". Given that two functions can agree on an **interval** in the real line and still be completely different functions, there is no reason that agreement on finitely many values (in fact, only values that can be represented in floating-point arithmetic) would imply some deeper structural link between the function and the interpolant.

Although the new question about *learning* the data generation mechanism has not been precisely defined, it is in a sense falsifiable and, therefore, worth considering further. So let's pretend we're in a toxic relationship with the Chebyshec polynomials: we tell them one thing and we mean another. Initially, we asked them to approximate the function $$f$$ on the interval $$I=[-1,1]$$, but now we expect them to behave like the function $$f$$ also **outside** the interval $$I$$. Extrapolation is an intuitively appealing concept of "learning", but, also, intellectually dishonest in this case. The domain of a function is part of its definition&mdash;there are no two ways about it. This is how the Chebyshev interpolant, built out of $$24$$ nodes, behaves on $$J\triangleq[-1.1,1.1]$$:

![Chebyshev extrapolation](/assets/snips/runges_phenomenon/cheb_extrapolation.png)

Nothing to see here, really. We fed the Chebyshev machinery a function $$f:I\mapsto\mathbb{R}$$ and the machinery did an excellent job at approximating it, and now we are checking the output on a **different** function $$\tilde{f}:J\mapsto\mathbb{R}$$.

But communicating intent clearly is important in most walks of life (probably not in politics or in environments dominated by politics). Then, let's reframe everything: we agree now that we want some kind of approximation that, well, approximates adequately the points we use to build the approximation, but, also, we want the approximating object to do a decent job when presented with new data not included in the construction of the approximation. This approach will offer a satisfactory illusion of "learning". The way to assess performance on this new task is called "cross-validation".

## Polynomial regression with cross-validation

Now we'll revert back to the standard basis $$1, X, X^2, \dots$$ of $$R[X]$$, although other bases can be superior depending on the task at hand.

*To be continued*

<!-- ## Another classical solution to the original problem

## Another problem -->