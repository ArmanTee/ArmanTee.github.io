---
layout: post
title:  "MCMC with q Part I: Inversion and Rejection Sampling"
short: "MCMC with kdb+ part I"
date:   2020-07-07
excerpt: "First in a 3-part series on Markov chain Monte-carlo sampling with q. This blog looks at two basic sampling methods used by modern statistical software to draw samples from basic probability densities. "
tag:
- kdb 
- sampling
- MCMC
- test
- jekyll
comments: true
---
*In this blog, we talk about sampling with kdb/q — we demonstrate some code to use to be able to sample from any appropriate univariate distribution using various Monte-Carlo methods.*

## 1 Intro

As a language, q is extremely efficient when it comes to working with vector based data structures, and is a natural choice for all sorts of data analysis, generally far-outranking other languages when it comes to speed. Albeit recently the number of available public libraries for kdb has expanded (see [https://github.com/KxSystems](https://github.com/KxSystems)), often end-users find that they must create their own tools to solve their problems. One of these problems is Monte-Carlo integration. 

In this blog series,  we take a look at building out some code to use q to sample from distributions through a number of methods. 

### Why Sample?

In statistics, often our goal is to compute some summary of a distribution model; these summaries are essentially integrals of probability densities.

The definition of the **mean** and the **variance** of a distribution is given by the formulae:

$$\mu_{x}  = \int_{x \in S} x \times f(x) dx;$$

$$\sigma_{x}^2  = \int_{x \in S} (x-\mu_x)^2 \times f(x) dx;$$

where $f(x)$ is the distribution at hand. 

 In many circumstances, these summaries have closed form solutions, meaning we can analytically solve and then calculate our desired values. Take for example the Beta distribution:

$$ f(x | \alpha,\beta) = \frac{\Gamma(\alpha +\beta)}{\Gamma(\alpha)\Gamma(\beta)}x^{\alpha - 1}(1 -x)^{\beta - 1}$$

where $\alpha$  and \($\beta$\) are the shape parameters of the function, and $\Gamma$ is the gamma function applied to the parameters.

Analytically, we can solve to find the mean and variance:

$$E(x | \alpha,\beta) = \frac{\alpha}{\alpha + \beta}$$

$$Var(x | \alpha,\beta) = \frac{\alpha\beta}{(\alpha + \beta)^2  (\alpha + \beta +1) }$$

but, what if we had a function that we couldn't solve analytically?  In cases like these, we turn to **sampling**. The logic of sampling is that we can generate (*simulate*) a sample of size  $n$  from the distribution of interest and then use discrete formulas applied to these samples to approximate the integrals of interest, i.e the above equations become:

$$\mu_{x}  = \int_{x \in S} x \times f(x) dx \approx{\frac{1}{n} \sum_{x \in S} x}$$

$$\sigma_{x}^2  = \int_{x \in S} (x-\mu_x)^2 \times f(x) dx \approx{\frac{1}{n} \sum_{x \in S} (x-\mu_x)^2}$$

### Random Number Generation

In order to use the simulation-based techniques described in this blog, we will need to be able to generate sequences of random numbers from the uniform distribution. To do this in q, we can use the `?[x;y]` overload. Where `x` is the number of samples and `y` is the maximum value of a single sample. A negative `x` is interpreted as *generate a random number without replacement.* The result also depends on the datatype of `y`, i.e

```q
// Generate 10 longs
q)10 ? 20
14 6 7 4 6 10 9 9 10 18

// Generate 10 floats
q)10 ? 20f
17.79407 2.139848 8.444447 15.34972 17.70322 8.717351 1.55764 8.467533 12.45664 3.944245

// Generate 10 longs without replacement
q)-10?10
7 1 3 9 5 0 4 6 8 2
// Cannot do the following:
q)-11?10
'length

q)-1?10.
'type
```

### Pseudo Random Number Generator (PRNG)

Along with most other analytics software, q does not generate genuine random numbers. Instead, pseudo-random numbers are generated that appear to be random but are actually deterministic. The implications of this is a topic of it's own, but it's important to understand that we can choose the initial seed via the command line parameter `\S`. This displays or sets the initial seed for the PRNG. Resetting the seed to the default value allows the pseudo-random values to be reproduced, i.e:

```q

q)a: 1000?10
q)\S
-314159i
// Resetting \S
q)\S -314159i
q)b: 1000?10
q)a~b
1b
```

## 2