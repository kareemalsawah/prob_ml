import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike

from expo_family import ExponentialFamily


class Poisson(ExponentialFamily):
    """
    Implementation of the Poisson distribution.
    """
    def __init__(self) -> None:
        super().__init__()

    def log_base_measure(self, x: ArrayLike | jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x)
        return -jax.scipy.special.gammaln(x[..., 0] + 1)

    def log_partition(self, params: ArrayLike | jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(jnp.asarray(params))[..., 0]

    def sufficient_statistics(self, x: ArrayLike | jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(x)

    def params_to_natural_params(self, params: ArrayLike | jnp.ndarray) -> jnp.ndarray:
        return jnp.log(jnp.asarray(params))

    def conjugate_log_partition(self, alpha: ArrayLike | jnp.ndarray, nu: ArrayLike | jnp.ndarray) -> jnp.ndarray:
        return jax.scipy.special.gammaln(alpha + 1) - (alpha + 1)*jnp.log(nu)


if __name__ == "__main__":
    # poisson_dist = Poisson()
    # lambdas = jnp.arange(1, 10)

    # ks = jnp.arange(0,20)

    # # plot the pdf
    # preds = jnp.exp(poisson_dist.log_pdf(ks[...,None], lambdas[..., None, None]))
    # # print(preds.shape)
    # plt.title("Poisson distribution")
    # for lam in lambdas:
    #     plt.plot(ks, jnp.exp(poisson_dist.log_pdf(ks[...,None], [lam])), "o-", ms=3,color='red', alpha=0.9)
    # plt.show()

    likelihood = Poisson()
    prior = likelihood.conjugate_prior()

    prior_natural_params = [1, 1] # alpha, nu
    data = [5]
    
    posterior = prior

    posterior_natural_params = likelihood.posterior_parameters(prior_natural_params, data)

    lambdas = jnp.linspace(0.1, 10, 100)
    plt.title("Bayesian inference with poisson and gamma")
    plt.plot(lambdas,
             jnp.exp(prior.log_pdf(lambdas[...,None], prior_natural_params)),
             "-",
             color="gray",
             label="prior",
             alpha=0.9)
    plt.plot(lambdas,
             jnp.exp(likelihood.log_pdf(data, lambdas[...,None])),
             "-",
             color="blue",
             label="likelihood",
             alpha=0.9)
    plt.plot(lambdas,
             jnp.exp(posterior.log_pdf(lambdas[...,None], posterior_natural_params)),
             "-",
             color="red",
             label="posterior",
             alpha=0.9)
    plt.legend()
    plt.show()