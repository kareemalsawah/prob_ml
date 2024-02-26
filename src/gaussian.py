"""
Gaussian Distribution Class
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike


class Gaussian:
    def __init__(
        self, mu: ArrayLike | jnp.ndarray, sigma: ArrayLike | jnp.ndarray, /
    ) -> jnp.ndarray:
        self.mu = mu
        self.sigma = sigma

    @functools.cached_property
    def L(self):
        """
        Cholesky decomposition of the covariance matrix (sigma)
        """
        return jax.linalg.cholesky(self.sigma)

    @functools.cached_property
    def L_factor(self):
        """
        Cholesky factor of the covariance matrix (sigma)
        """
        return jax.scipy.linalg.cho_factor(self.sigma, lower=True)

    @functools.cached_property
    def log_det(self):
        """
        Log determinant of the covariance matrix
        """
        return 2 * jnp.sum(jnp.log(jnp.diag(self.L)))

    @functools.cached_property
    def prec(self):
        """
        Precision: Inverse of the covariance matrix
        Might be unstable due to the matrix inverse
        """
        return jnp.linalg.inv(self.sigma)

    def prec_mult(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Right multiply by the precision
        """
        return jax.scipy.linalg.cho_solve(self.L_factor, x)

    @functools.cached_property
    def mp(self):
        """
        Precision adjusted mean
        """
        return self.prec_mult(self.mu)

    def pdf(self, data: ArrayLike | jnp.ndarray, /) -> jnp.ndarray:
        """Signature (D)->()
        Evalute the pdf of this gaussian
        """
        return jnp.exp(self.log_pdf(data))

    def log_pdf(self, data: ArrayLike | jnp.ndarray, /) -> jnp.ndarray:
        """Signature (D)->()
        Evalute the log pdf of this gaussian
        """
        d = data.shape[-1]
        assert self.mu.shape[0] == d

        diff = data - self.mu

        log_pdf = -0.5 * (diff.T @ self.prec_mult(diff))

        log_partition = 0.5 * self.log_det + d / 2 * jnp.log(2 * jnp.pi)

        return log_pdf - log_partition

    def __mult__(self, other):
        """
        Product of two gaussian pdfs over the same random variable is also gaussian
        """
        new_sigma = jnp.linalg.inv(self.prec + other.prec)
        new_mu = new_sigma @ (self.mp + other.mp)
        return Gaussian(new_mu, new_sigma)

    def __rmatmul__(self, A):
        """
        Multiplying by a matrix A
        """
        return Gaussian(A @ self.mu, A @ self.sigma @ A.T)

    def __add__(self, other):
        """
        Add another gaussian or scalar
        """
        if isinstance(other, Gaussian):
            return Gaussian(self.mu + other.mu, self.sigma + other.sigma)

        return Gaussian(self.mu + other, self.sigma)

    def condition(self, A, y, Lambda):
        """
        Condition to get p(x|Ax=y) where y has Covariance Lambda
        """
        gram = A @ Lambda @ A.T + Lambda
        L = jax.scipy.linalg.cho_factor(gram, lower=True)
        new_mean = self.mu + self.sigma @ A.T @ jax.scipy.linalg.cho_solve(
            L, y - A @ self.mu
        )
        new_sigma = self.sigma + self.sigma @ A.T @ jax.scipy.linalg.cho_solve(
            L, A @ self.sigma
        )
        return Gaussian(new_mean, new_sigma)

    @functools.cached_property
    def std(self):
        """
        Returns the std of all variables
        """
        return jnp.sqrt(jnp.diag(self.sigma))

    def sample(self, key, n_samples: int) -> jnp.ndarray:
        """
        Generate samples from a multivariate gaussian
        """
        return jax.random.multivariate_normal(
            key, mean=self.mu, cov=self.sigma, shape=(n_samples,), method="svd"
        )
