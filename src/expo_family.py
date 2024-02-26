"""
Base class for exponential family distributions
"""

from __future__ import annotations

import abc
import functools

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike


class ExponentialFamily(abc.ABC):
    @abc.abstractmethod
    def sufficient_statistics(self, x: ArrayLike | jnp.ndarray, /) -> jnp.ndarray:
        """Signature `(D)->(P)`
        Sufficient statistics of the exponential family distribution

        Parameters
        ----------
        x: jnp.ndarray
            Data, shape=(-1, d)

        Returns
        -------
        sufficient_statistics: jnp.ndarray
            shape=(-1, p)
        """

    @abc.abstractmethod
    def log_partition(self, params: jnp.ndarray, /) -> jnp.ndarray:
        """Signature `(P)->()`
        Log partition function of the exponential family distribution

        Parameters
        ----------
        params: jnp.ndarray
            Natural parameters, shape=(-1, p)

        Returns
        -------
        log_partition: jnp.ndarray
            shape=(-1,)
        """

    @abc.abstractmethod
    def log_base_measure(self, x: ArrayLike | jnp.ndarray, /) -> jnp.ndarray:
        """Signature `(D)->()`
        Log base measure of the exponential family distribution

        Parameters
        ----------
        x: jnp.ndarray
            Data, shape=(-1, d)

        Returns
        -------
        log_base_measure: jnp.ndarray
            shape=(-1,)
        """

    def params_to_natural_params(self, params: ArrayLike | jnp.array, /) -> jnp.ndarray:
        """Signature `(P)->(P)`
        Converts canonical parameters to natural parameters
        By default, the canonical parameters are the natural parameters

        Parameters
        ----------
        params: jnp.ndarray
            Canonical parameters, shape=(-1, p)

        Returns
        -------
        natural_params: jnp.ndarray
            Natural parameters, shape=(-1, p)
        """
        return jnp.asarray(params)

    def log_pdf(self, x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        """Signature `(D, P)->()`
        Log probability density function of the exponential family distribution

        Parameters
        ----------
        x: jnp.ndarray
            Data, shape=(-1, d)
        params: jnp.ndarray
            Canonical parameters

        Returns
        -------
        log_pdf: jnp.ndarray

        """
        x = jnp.asarray(x)
        log_base_measure = self.log_base_measure(x)
        natural_parameters = self.params_to_natural_params(params)
        linear_term = (
            self.sufficient_statistics(x)[..., None, :] @ natural_parameters[..., None]
        )
        log_partition = self.log_partition(natural_parameters)
        return linear_term[..., 0, 0] - log_partition + log_base_measure

    def conjugate_log_partition(
        self, alpha: ArrayLike | jnp.array, nu: ArrayLike | jnp.array, /
    ) -> jnp.array:
        """Signature `(P),()->()`
        Log partition function of the conjugate to this exponential family distribution
        """
        raise NotImplementedError()

    def conjugate_prior(self) -> ConjugateFamily:
        """Returns the conjugate prior for this exponential family distribution"""
        return ConjugateFamily(self)

    def posterior_parameters(
        self,
        prior_natural_params: ArrayLike | jnp.ndarray,
        data: ArrayLike | jnp.ndarray,
    ) -> jnp.ndarray:
        """Signature `(P),(D)->(P)`"""
        prior_natural_params = jnp.asarray(prior_natural_params)
        sufficient_statistics = self.sufficient_statistics(data)

        n = sufficient_statistics[..., 0].size
        expected_sufficient_statistics = jnp.sum(
            sufficient_statistics, axis=(tuple(range(sufficient_statistics.ndim)))
        )

        alpha_prior, nu_prior = prior_natural_params[:-1], prior_natural_params[-1]

        return jnp.append(alpha_prior + expected_sufficient_statistics, nu_prior + n)


class ConjugateFamily(ExponentialFamily):
    def __init__(self, likelihood: ExponentialFamily) -> None:
        self._likelihood = likelihood

    @functools.partial(jnp.vectorize, excluded={0}, signature="(d)->(p)")
    def sufficient_statistics(
        self, likelihood_canonical_params: ArrayLike | jnp.ndarray, /
    ) -> jnp.ndarray:
        """Signature `(D)->(P)`
        Contain the natural parameters and negative log partition function
        of the likelihood distribution
        """
        w = self._likelihood.params_to_natural_params(likelihood_canonical_params)
        return jnp.append(w, -self._likelihood.log_partition(w))

    def log_base_measure(self, w: ArrayLike | jnp.ndarray, /) -> jnp.ndarray:
        w = jnp.asarray(w)
        return jnp.zeros_like(w[..., 0])

    def log_partition(self, natural_params: ArrayLike | jnp.ndarray, /) -> jnp.ndarray:
        """Signature `(P)->()`"""
        natural_params = jnp.asarray(natural_params)

        alpha, nu = natural_params[:-1], natural_params[-1]
        return self._likelihood.conjugate_log_partition(alpha, nu)

    def unnormalized_log_pdf(
        self,
        likelihood_canonical_params: ArrayLike | jnp.ndarray,
        natural_params: ArrayLike | jnp.ndarray,
        /,
    ) -> jnp.ndarray:
        """Signature `(D),(P)->()`"""
        return self.sufficient_statistics(likelihood_canonical_params) @ jnp.asarray(
            natural_params
        )

    def laplace_precision(
        self, natural_params: ArrayLike | jnp.ndarray, mode: ArrayLike | jnp.ndarray, /
    ) -> jnp.ndarray:
        """Signature `(P),(D)->()`"""
        return -1 * jax.hessian(self.unnormalized_log_pdf, argnums=0)(
            jnp.asarray(mode), natural_params
        )
