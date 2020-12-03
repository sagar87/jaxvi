from abc import abstractmethod
from jaxvi.abstract import ABCMeta, abstract_attribute
import jax.numpy as jnp
from jax.scipy.stats import norm, gamma


class Model(metaclass=ABCMeta):
    @abstract_attribute
    def latent_dim(self):
        pass

    @abstractmethod
    def inv_T(self, zeta: jnp.DeviceArray) -> jnp.DeviceArray:
        pass

    @abstractmethod
    def log_joint(self, theta: jnp.DeviceArray) -> jnp.DeviceArray:
        pass


class LinearRegression(Model):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.latent_dim = x.shape[1] + 1

    def inv_T(self, zeta: jnp.DeviceArray) -> jnp.DeviceArray:
        return jnp.append(zeta[:-1], jnp.exp(zeta[-1]))

    def log_joint(self, theta: jnp.DeviceArray) -> jnp.DeviceArray:
        betas = theta[:2]
        sigma = theta[2]

        beta_prior = norm.logpdf(betas, 0, 10).sum()
        sigma_prior = gamma.logpdf(sigma, a=1, scale=2).sum()
        yhat = jnp.inner(self.x, betas)
        likelihood = norm.logpdf(self.y, yhat, sigma).sum()

        return beta_prior + sigma_prior + likelihood
