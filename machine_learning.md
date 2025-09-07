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
["Runge's phenomenon"](https://en.wikipedia.org/wiki/Runge's_phenomenon) is an observation related to the interpolation of a function by polynomials using equidistant nodes. Today, the observation is somewhat trivial or, at least, mainstream; however, it drew attention when it was first made, and survives to this day under the name "Runge's phenomenon", because the standards or rigour among physicists were quite lax back then. Even mathematics, had only recently started getting serious with the help of Cauchy and Weierstrass. And since Dieudonné has written about Runge's phenomenon, it's impossible to put it in better terms (Calcul Infinitésimal, p. 16):

![Dieudonné, Calcul Infinitésimal](/assets/snips/runges_phenomenon/dieudonne_runge_phenomenon.png)

Loosely translated:

> But where physicists venture into dangerous territory is when they tend to accept as "obvious" what is no longer so at all, and forget that our "intuition" is a rather rudimentary tool that grossly deceives us on occasion. Contrary to what many of them believe, it is not necessary, to challenge results they accept without discussion, to seek out functions as "monstrous" as functions without derivatives; the "phenomenon of Runge" (chap. IX, Appendix) shows that the classical polynomial interpolation procedure can very well diverge for analytic functions as "excellent" as one could desire; and there are analytic functions for `|z| < 1`, continuous throughout the disk `|z| ≤ 1`, and yet which transform the circle `|z| = 1` into a Peano curve filling a square.

Because there are many moving pieces in what follows, it is useful to have from the beginning a *conceptual* summary of what Runge's phenomenon is about: it simply says that the *obvious* choice of interpolating polynomials can be a terrible idea. It's not a hard problem, let alone an unsolved one. Learning a bit about it, however, can be informative.

## Runge's function
First, let's define Runge's function, which is the function he used as an example in his [paper](https://archive.org/details/zeitschriftfrma12runggoog/page/224/mode/2up?view=theater) on interpolation at equidistant points. But, to make things a bit more dramatic, let's add a coefficient of `10` in the denominator:

![Runge's example function](/assets/snips/runges_phenomenon/runges_function_original.png)

```python
def ground_truth(x: np.array, noise: bool = False, seed: int = None) -> pd.DataFrame:

    if noise:
        if seed is None:
            raise ValueError('If `noise is True` then `seed` cannot be `None`.')
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

That the graphs of the absolute and relative errors look identical is a coincidence due to the range of values of $$x$$ and $$y$$. No need to worry about that. There is no sleight of hand here; Chebyshev polynomials do work extremely well for certain tasks, and it's an entire rabbit hole to go down if one is willing to. An excellent book of applied mathematics that demonstrates some powerful techniques, especially for root finding, is [J. Boyd's book](https://www.google.ca/books/edition/Solving_Transcendental_Equations/29CgBAAAQBAJ?hl=en&gbpv=0). There is also the whole "chebfun" business that spun out of these ideas &mdash; google is your friend.

## Another classical solution to the problem

## Another problem