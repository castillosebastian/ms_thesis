# Generative Model

Generative models are a class of statistical models that aim to learn the underlying data distribution from a given dataset. These models provide a way to generate new samples that are statistically similar to the training data. They have gained substantial attention in various domains, such as image generation, speech synthesis, and even drug discovery.

Given a dataset of observed samples, one starts by selecting a distributional model parameterized by $(\theta)$. The objective is to estimate $(\theta)$ such that it aligns optimally with the observed samples.The anticipation is that it can also generalize to samples outside the training set.

The optimal distribution is hence the one that maximizes the likelihood of producing the observed data, giving lower probabilities to infrequent observations and higher probabilities to the more common ones (the principle underlying this assumption is: 'the world is a boring place' Bhiksha Raj).

### The Challenge of Maximum Likelihood Estimates (MLE) for Unseen Observations

When training generative models, a natural objective is to optimize the model parameters such that the likelihood of the observed data under the model is maximized. This method is known as **Maximum Likelihood Estimation (MLE)**. In mathematical terms, given observed data $X$, the MLE seeks parameters $\theta$ that maximize:

> $p_\theta(X)$

However, for many generative models, especially those that involve latent or unobserved variables, the likelihood term involves summing or integrating over all possible configurations of these latent variables. Mathematically, this turns into:

> $p_\theta(X) = \sum_{Z} p_\theta(X,Z)$\
> or\
> $p_\theta(X) = \int p_\theta(X,Z) dZ$

Computing the log-likelihood, which is often used for numerical stability and optimization ease, leads to a log of summations (for discrete latent variables) or a log of integrals (for continuous latent variables):

> $log p_\theta(X) = \log \sum_{Z} p_\theta(X,Z)$\
> or\
> $log p_\theta(X) = \log \int p_\theta(X,Z) dZ$

These expressions are typically intractable to optimize directly due to the presence of the log-sum or log-integral operations. Where do these formulas come frome? They come forom *marginalization* in the context of joint probability.

### Marginalization in the Context of Joint Probability

When discussing the computation of the joint probability for observed and missing data, the term "marginalizing" refers to summing or integrating over all possible outcomes of the missing data. This process provides a probability distribution based solely on the observed data. For example, let's assume:

-   $X$ is the observed data
-   $Z$ is the missing data
-   The joint probability for both is represented as $p(X,Z)$

If your primary interest lies in the distribution of $X$ and you wish to eliminate the dependence on $Z$, you'll need to carry out marginalization for $Z$. For discrete variables, the marginalization involves the logarithm of summation:

$$ \log \left( \sum_{z} p(X,Z=z) \right),$$

for continuous variables, it pertains to integration:

$$ \int p(X,Z) dZ $$

However, it's essential to note that these functions includes the log of a sum, which defies direct optimization. Can we get an approximation to this that is more tractable (without a summation or integral within the log)?

### Overcoming the Challenge with Expectation Maximization (EM)

To address the optimization challenge in MLE with latent variables, the **Expectation Maximization (EM)** algorithm is employed. The EM algorithm offers a systematic approach to iteratively estimate both the model parameters and the latent variables.

The algorithm involves two main steps:

1.  **E-step (Expectation step)**: involves computing the expected value of the complete-data log-likelihood with respect to the posterior distribution of the latent variables given the observed data.
2.  **M-step (Maximization step)**: Update the model parameters to maximize this expected log-likelihood from the E-step.

By alternating between these two steps, EM ensures that the likelihood increases with each iteration until convergence, thus providing a practical method to fit generative models with latent variables.

For E-step the **Variational Lower Bound** is used. Commonly referred to as the Empirical Lower BOund (ELBO), is a central concept in variational inference. This method is used to approximate complex distributions (typically posterior distributions) with simpler, more tractable ones. The ELBO is an auxiliary function that models MLE by iteratibly maximize a 'lower bound' function

Variational inference is set up as an optimization problem where the objective is to find a distribution $q$ from a simpler family of distributions that is close to the target distribution $p$. The closeness is measured using the Kullback-Leibler (KL) divergence. The ELBO serves as the objective function to be maximized in this optimization.

The ELBO can be derived from the logarithm of the marginal likelihood of the observed data. Given:

> $p(\mathbf{x})$ is the marginal likelihood of the observed data $\mathbf{x}$,
>
> $q(\mathbf{z})$ is the variational distribution over the latent variables $\mathbf{z}$,
>
> $p(\mathbf{z} | \mathbf{x})$ is the true posterior distribution of $\mathbf{z}$.

The ELBO is given by:

${ELBO}(q) = \mathbb{E}_q[\log p(\mathbf{x}, \mathbf{z})] - \mathbb{E}_q[\log q(\mathbf{z})]$

Where:

$\mathbb{E}_q$ denotes the expectation with respect to the distribution $q$. The first term on the right-hand side measures how well the joint distribution $p(\mathbf{x}, \mathbf{z})$ is modeled by $q$. -The second term is the entropy of $q$, which acts as a regularizer. Maximizing the ELBO is equivalent to minimizing the KL divergence between the variational distribution $q$ and the true posterior $p(\mathbf{z} | \mathbf{x})$. Intuitively, by maximizing the ELBO, you're trying to find a balance between a distribution $q$ that closely matches the target distribution $p$ (as measured by the joint likelihood) and one that maintains uncertainty (as measured by its entropy).

In SUMMARY, generative models offer a powerful approach to understanding and generating data. The challenges posed by the MLE in the presence of latent variables are effectively addressed by the EM algorithm, making it a cornerstone method in the training of many generative models.