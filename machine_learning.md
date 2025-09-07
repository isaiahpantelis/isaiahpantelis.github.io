---
layout: default
title: Machine Learning
permalink: /machine_learning
---

<!-- *Under construction* -->

#### CONTENTS
- [An old problem with an old solution](#an-old-problem-with-an-old-solution)

<hr>

# An old problem with an old solution
["Runge's phenomenon"](https://en.wikipedia.org/wiki/Runge's_phenomenon) is an observation related to the interpolation of a function by polynomials using equidistant nodes. Today, the observation is somewhat trivial or, at least, mainstream, but it drew attention when it was first made and survives to this day under the name "Runge's phenomenon" because of the standards or rigour have not always been the same among physicists and applied mathematicians. And since Dieudonné has written about Runge's phenomenon, it's impossible to put it in better terms ():

![Dieudonné, Calcul Infinitésimal](/assets/snips/runges_phenomenon/dieudonne_runge_phenomenon.png)

Loosely translated:

> But where physicists venture into dangerous territory is when they tend to accept as "obvious" what is no longer so at all, and forget that our "intuition" is a rather rudimentary tool that grossly deceives us on occasion. Contrary to what many of them believe, it is not necessary, to challenge results they accept without discussion, to seek out functions as "monstrous" as functions without derivatives; the "phenomenon of Runge" (chap. IX, Appendix) shows that the classical polynomial interpolation procedure can very well diverge for analytic functions as "excellent" as one could desire; and there are analytic functions for `|z| < 1`, continuous throughout the disk `|z| ≤ 1`, and yet which transform the circle `|z| = 1` into a Peano curve filling a square.

Because there are many moving pieces in what follows, it is useful to have from the beginning a *conceptual* summary of what Runge's phenomenon is about: it simply says that the *obvious* choice of interpolating polynomials can be a terrible idea. It's not a hard problem, let alone an unsolved one. Learning a bit about it, however, can be informative.

## Runge's function
First, let's define Runge's function, which is the function he used as an example in his [paper](https://archive.org/details/zeitschriftfrma12runggoog/page/224/mode/2up?view=theater) on interpolation at equidistant points. But, to make things a bit more dramatic, let's add a coefficient of `10` in the denominator:

![Runge's example function](/assets/snips/runges_phenomenon/runges_function_original.png)

```python
def ground_truth(x, noise=False, seed=None):

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