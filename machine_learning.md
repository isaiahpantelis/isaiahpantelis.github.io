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
["Runge's phenomenon"](https://en.wikipedia.org/wiki/Runge's_phenomenon) is an observation, published by Carl Runge in 1901, related to the interpolation of a function by polynomials using equidistant nodes.[^1] Today, the observation is somewhat trivial or, at least, mainstream; however, it drew attention when it was first made because the standards or rigour among physicists were quite lax back then. Even mathematics had only recently started adopting a grown up attitude towards rigour with the help of Cauchy and Weierstrass. But, if Dieudonné has written about a topic, there is no point in trying to do better. And he *has* made a remark about Runge's phenomenon (Calcul Infinitésimal, p. 16):

![Dieudonné, Calcul Infinitésimal](/assets/snips/runges_phenomenon/dieudonne_runge_phenomenon.png)

Loosely translated:

> But where physicists venture into dangerous territory is when they tend to accept as "obvious" what is no longer so at all, and forget that **our "intuition" is a rather rudimentary tool that grossly deceives us on occasion**.[^2] Contrary to what many of them believe, it is not necessary, to challenge results they accept without discussion, to seek out functions as "monstrous" as functions without derivatives; the "phenomenon of Runge" (chap. IX, Appendix) shows that the classical polynomial interpolation procedure can very well diverge for analytic functions as "excellent" as one could desire; and there are analytic functions for $${\\|}z{\\|}\lt 1$$, continuous throughout the disk $${\\|}z{\\|}\leq 1$$, and yet which transform the circle $${\\|}z{\\|} = 1$$ into a Peano curve filling a square.

Here is a *conceptual* summary of what Runge's phenomenon is about: it simply says that the **obvious** choice of interpolating polynomials can be a terrible idea. What it does **not** say is say that polynomials are useless for approximation. Runge's phenomenon is not a hard problem to adress, let alone an unsolved one. Learning about it, however, can be informative.

## Runge's function
First, let's define Runge's function, which is the function he used as an example in his [paper](https://archive.org/details/zeitschriftfrma12runggoog/page/224/mode/2up?view=theater) on interpolation at equidistant points. But, to make things a bit more dramatic, we'll add a coefficient of $$10$$ in the denominator:

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

*(Since we're playing around with interpolation, it makes sense to have the option to add noise on a whim. Hence the extra lines of code.)*

## The problem
If you're all wide-eyed when it comes to approximation theory, the following may seem counter-intuitive.

Suppose that 

$$x_k, \quad k=0,\ldots,n-1, \quad x_k\in[0,1]$$

are equidistant points in $$[0,1]$$ (aka nodes) and $$y_k\triangleq f(x_k)$$ are the corresponding values of the Runge function. Then, the unique polynomial $$p_{n-1}^L$$ of degree $$n-1$$ that passes through the points $$(x_k,y_k)$$ becomes a *terrible approximation* (this is a technical term) of the function $$f$$ as $$n$$ increases:

![Lagrange interpolating polynomial for the Runge function](/assets/snips/runges_phenomenon/lagrange_approx_runge.png)

## A classical solution to the problem

There is a classical well-known remedy to this problem: **choose your nodes wisely**. Specifically, since all truths in the real line pass through the complex plane (according to Hadamard), choose the nodes to be projections of equidistant points (the distance being measured by arc length) on the unit circle. In other words, a better choice is the *Chebyshev points of the 1st kind*. 

![Chebyshev interpolating polynomial for the Runge function](/assets/snips/runges_phenomenon/chebyshev_approx_runge.png)

Because eyeballing pixels when it comes to approximation errors can be extremely useful but also a fool's errand (and you don't know which applies when), here are also the errors of the Chebyshev interpolation:

![Chebyshev interpolation errors](/assets/snips/runges_phenomenon/cheb_interp_errors.png)

That the graphs of the absolute and relative errors look identical is a coincidence due to the range of values of $$x$$ and $$y$$. No need to worry about that here. But the fact that even the relative zero approaches zero, although the function values tend to zero is worth appreciating. 

There is no sleight of hand &mdash; Chebyshev polynomials **do** work extremely well for certain tasks; and it's an entire rabbit hole to go down if one is willing to. An excellent book of applied mathematics that demonstrates some powerful techniques, especially for root finding, is [J. Boyd's book](https://www.google.ca/books/edition/Solving_Transcendental_Equations/29CgBAAAQBAJ?hl=en&gbpv=0). There is also the whole "chebfun" business that spun out of these ideas (google is your friend). The overarching principle is that "continuous mathematical objects" (whatever that means) can be replaced or proxied up to machine precision by their Chebyshev approximations. But there is more magic to it, actually: from the proxy one can compute roots, derivatives, and integrals. The major limitation of this paradigm is the inability to handle high-dimensional problems, but enough said for now (and, no, "tensor products" have not fully addressed the limitation).

## A different problem
At this point we know that:

1. Exact interpolation at equidistant points *can* be a terrible idea.
2. Exact interpolation at cleverly sampled points *can* be a fantastic idea.

It all depends on the task at hand. But one can ask **other** interesting questions. For example, is a good interpolant, such as the one provided by the Chebyshev interpolation, somehow inherently tied to the function that generated the interpolated values? Or, put differently, is a good interpolant "learning" the data generation mechanism? Now, a mathematically literate person (say, someone who has been exposed to post 18-th century mathematics)[^3] should immediately offer an emphatic "no" or an emphatic "no, unless you prove otherwise". Given that two functions can agree on an **interval** in the real line and still be completely different functions, there is no reason that agreement on finitely many values[^4] would imply some deeper structural link between the function and the interpolant.

Although the newly formulated question about *learning* the data generation mechanism (as oppposed to finding a polynomial that passes through some points) has not been precisely defined, it is in a sense falsifiable and, therefore, worth considering further. So let's pretend we're in a toxic relationship with the Chebyshec polynomials: we tell them one thing but we mean another. Initially, we had asked of them to approximate the function $$f$$ on the interval $$I=[-1,1]$$, but now we expect them to behave like the function $$f$$ also **outside** the interval $$I$$. This is an instance of "extrapolation" and may seem like an intuitively appealing concept of "learning". After all, if you are actually learning the **function**, then you'd behave like the function no matter what, no?. But intuitive appeal or not, "the first principle is that you must not fool yourself"[^5]. The domain of a function is part of the function's definition&mdash;there are no two ways about it&mdash;and we had asked for an approximation on $$[-1,1]$$ specifically. A different domain means a different function.[^6]

This is how the Chebyshev interpolant, built out of $$24$$ nodes, behaves on $$J\triangleq[-1.1,1.1]$$:[^7]

![Chebyshev extrapolation](/assets/snips/runges_phenomenon/cheb_extrapolation.png)

Nothing to see here, really. Again, We fed the Chebyshev machinery a function $$f:I\mapsto\mathbb{R}$$ and the machinery did an excellent job at approximating it. But now we are checking the constructed approximation against a different function $$\tilde{f}:J\mapsto\mathbb{R}$$.

But communicating intent clearly is important in most walks of life.[^8] So, then, let's reframe everything: let's agree now that what we *really* want is *some* kind of approximation that, well, approximates adequately the points used to build the approximation itself, but, also, we want the approximating object, in this case the interpolant, to do a decent job when presented with **new data never seen before**. This approach may offer a satisfactory illusion of "learning the data generation mechanism". The way to assess performance on this new task is usually called "cross-validation" (CV).

## Polynomial regression with cross-validation

To tackle the new task, we shift mindset and methodology. The tools in the previous sections were from classical approximation theory. The tools in this section have a throw-the-kitchen-sink-at-the-problem flavour. That is, we'll use some **rudimentary machine learning**. First, we'll revert to the standard basis[^9] $$1, X, X^2, \dots$$ of $$R[X]$$ and use an `sklearn` preprocessor to create polynomial features. Then, we'll generate a ton of data, instead of a handfull of nodes, and use cross-validation to find the best approximating polynomial by doing a grid search over the polynomial's degree. Specifically, let's use $$1,000$$ points $$(x_k,f(x_k))$$, where the $$x_k$$ are equidistant in $$I=[-1,1]$$ and $$f:I\mapsto\mathbb{R}$$ is the Runge function, and 10-fold cross-validation. Also, we will search over degrees ranging from $$1$$ to $$100$$.

![Polynomial regression using 10-fold CV, 1,000 equidistant data points, and a grid search over degrees from 1 to 100](/assets/snips/runges_phenomenon/poly_regr_wo_shuffling.png)

The result is kind of interesting. Some observations:

1. Extrapolation to $$[-1.1, 1.1]$$ is still atrocious. Put differently, despite the use of 10-fold CV on the training set $$[-1,1]$$, performance on the test sets $$[-1.1,-1]$$ and $$[1,1.1]$$ is underwhelming at best. 
1. The best degree singled out by 10-fold CV, with the given choice of hyperparameters, number of data points, etc, is a lowly $$8$$.

Out of curiosity, we can repeat this last experiment using the same Chebyshev nodes as before:

![Polynomial regression using 10-fold CV, 1,000 Chebyshev nodes, and a grid search over degrees from 1 to 100](/assets/snips/runges_phenomenon/poly_regr_wo_shuffling_cheb_nodes.png)

This is an intuitive outcome: with $$1,000$$ Chebyshev nodes in $I$ and with the characteristic property that Chebyshev nodes have of clustering towards the endpoints of the interval, there isn't enough room for the approximating polynomial to start wiggling. At the same time, performance on the test set (i.e., extrapolation) is even worse than before.

But there is something else slightly subtle going on here as well that can be easy to miss without some prior familiarity with both the theory and implementation of ML algos. The constructor `KFold` has `False` as the default value for the argument `shuffle`; therefore, calling the constructor like this

```python
cv = KFold(n_splits=10)
```

will not permute the rows of the data frame that contains the training data. This can be confirmed by checking the indices of the training set:

![Indices of the training set without shuffling](/assets/snips/runges_phenomenon/poly_regr_wo_shuffling_test_idx.png){: style="width: 50%; margin-left: auto; margin-right: auto; display: block;"}

Quick sanity check: $$1,000$$ training data points with $$10$$ folds results in test sets of $$100$$ points to be used in the grid search. More importantly, the test sets are obtained by sliding a window from left to right[^10] inside $$[-1,1]$$.

Often this is not the desired behaviour, especially if one is using an ML method that assumes i.i.d. samples. Here, however, there is no statistical inference&mdash;we're simply solving a least squares problem.[^11]

The natural next thing to try is equidistant nodes + shuffling:

![Polynomial regression using 10-fold CV, 1,000 equidistant data points, and a grid search over degrees from 1 to 100 with shuffling](/assets/snips/runges_phenomenon/poly_regr_with_shuffling.png)

The result, in this case, is an interpolating polynomial of much higher degree $$(71)$$ that overfits. The conclusion is **definitely not** that a general rule that "shuffling leads to overfitting". The whole point of this exercise is simply to carefully consider various aspects of applying different techniques to the same problem.

<hr>
**Exercise:** Try $$2,000$$ Chebyshev nodes, add some noise, and using shuffling in the cross validation.
<hr>

#### FOOTNOTES

[^1]: In this note, the focus is on real-valued functions of a single real variable.
[^2]: The emphasis is mine.
[^3]: Among other things, 18th century mathematicians believed that all functions are real analytic. And they would fight you for it.
[^4]: In fact, only values that can be represented in floating-point arithmetic.
[^5]: [Cargo cult science](https://people.cs.uchicago.edu/~ravenben/cargocult.html)
[^6]: The fact that obvious relations can be defined between functions, such as a function $$f$$ being the **restriction** or the **continuation** of another function $$g$$ is a different story. One has to maintain clarity of thought.
[^7]: I won't even bother showing the extrapolation using the Lagrange polynomial. As you can guess, it's abysmal.
[^8]: Probably not in politics or trading.
[^9]: Although it's not the best basis for every task, familiarity wins.
[^10]: The training data are sorted when they are generated.
[^11]: That of polynomial regression.