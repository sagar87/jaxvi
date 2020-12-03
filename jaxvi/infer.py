from typing import Callable, NamedTuple, Tuple
import jax.numpy as jnp
from jax.scipy.stats import norm, multivariate_normal
from jax import jacfwd
from jax import grad
from jax.ops import index_update

from jaxvi.models import Model

# class ADVIState(NamedTuple):
#     phi: jnp.DeviceArray
#     grad_phi: jnp.DeviceArray


class ADVI(object):
    def __init__(self, model: Model):
        """
        A native implementation of ADVI.

        Arguments:
            log_joint: function to compute the log joint p(x, theta)
            inv_T: function to map latent parameters to the real space
            latent_dim: s
        """
        self.model = model
        self.latent_dim = model.latent_dim
        self.inv_T = model.inv_T
        self.jac_T = jacfwd(self.inv_T)

        # gradients
        self.grad_joint = grad(model.log_joint)
        self.grad_det_J = grad(self.log_abs_det_jacobian)

        # variational parameters
        self.phi = jnp.zeros(2 * model.latent_dim)

    # def __call__(self, eta):
    #     phi = jnp.zeros(2 * self.latent_dim)
    #     return ADVIState(phi, self.grad(eta, phi))

    def log_abs_det_jacobian(self, zeta: jnp.DeviceArray) -> jnp.DeviceArray:
        return jnp.log(jnp.abs(jnp.linalg.det(self.jac_T(zeta))))

    def mu(self, phi: jnp.DeviceArray) -> jnp.DeviceArray:
        return phi[: self.latent_dim]

    def sigma(self, phi: jnp.DeviceArray) -> jnp.DeviceArray:
        return phi[self.latent_dim :]

    def omega(self, phi: jnp.DeviceArray) -> jnp.DeviceArray:
        return jnp.exp(self.sigma(phi))

    def inv_S(self, eta: jnp.DeviceArray, phi: jnp.DeviceArray) -> jnp.DeviceArray:
        """
        Transforms eta to zeta.
        """
        return (eta * self.omega(phi)) + self.mu(phi)

    def variational_entropy(
        self, zeta: jnp.DeviceArray, phi: jnp.DeviceArray
    ) -> jnp.DeviceArray:
        probs = norm.pdf(zeta, loc=self.mu(phi), scale=self.omega(phi))
        return -(probs * jnp.log(probs)).sum()

    def grad(self, eta: jnp.DeviceArray, phi: jnp.DeviceArray) -> jnp.DeviceArray:
        """ Returns nabla mu and nabla omega """
        zeta = self.inv_S(eta, phi)
        theta = self.inv_T(zeta)

        # compute gradients
        grad_joint = self.grad_joint(theta)
        grad_inv_t = self.jac_T(zeta)
        grad_trans = self.grad_det_J(zeta)

        grad_mu = grad_inv_t @ grad_joint + grad_trans
        grad_omega = grad_mu * eta * self.omega(phi) + 1

        return jnp.append(grad_mu, grad_omega)

    def elbo(self, eta: jnp.DeviceArray, phi: jnp.DeviceArray) -> jnp.DeviceArray:
        zeta = self.inv_S(eta, phi)
        theta = self.inv_T(zeta)

        return (
            self.model.log_joint(theta)
            + self.log_abs_det_jacobian(zeta)
            + self.variational_entropy(zeta, phi)
        )


class FullRankADVI(ADVI):
    def __init__(self, model):
        """
        A naive implementation of ADVI.

        Arguments:
            log_joint: function to compute the log joint p(x, theta)
            inv_T: function to map latent parameters to the real space
            latent_dim: s
        """
        super().__init__(model)
        # variational parameters
        self.phi = jnp.append(
            jnp.zeros(self.latent_dim),
            jnp.ones(int(self.latent_dim * (self.latent_dim + 1) / 2)),
        )

    def L(self, phi: jnp.DeviceArray) -> jnp.DeviceArray:
        L = jnp.zeros((self.latent_dim, self.latent_dim))
        L = index_update(L, jnp.tril_indices(self.latent_dim), phi[self.latent_dim :])
        return L

    def inv_L(self, phi: jnp.DeviceArray) -> jnp.DeviceArray:
        L = self.L(phi)
        return jnp.linalg.inv(L @ L.T)

    def inv_S(self, eta: jnp.DeviceArray, phi: jnp.DeviceArray) -> jnp.DeviceArray:
        """
        Transforms eta to zeta.
        """
        return (self.L(phi) @ eta) + phi[: self.latent_dim]

    def variational_entropy(self, zeta, phi):
        L = self.L(phi)
        probs = multivariate_normal.pdf(zeta, mean=phi[: self.latent_dim], cov=L @ L.T)
        return -(probs * jnp.log(probs)).sum()

    def grad(self, eta, phi):
        """ Returns nabla mu and nabla omega """
        zeta = self.inv_S(eta, phi)
        theta = self.inv_T(zeta)

        # compute gradients
        grad_joint = self.grad_joint(theta)
        grad_inv_t = self.jac_T(zeta)
        grad_trans = self.grad_det_J(zeta)

        grad_mu = grad_inv_t @ grad_joint + grad_trans
        # print(grad_μ, η, grad_μ * η, grad_μ * η.T, self.inv_L(ϕ).T)
        grad_L = (grad_mu * eta + self.inv_L(phi).T)[jnp.tril_indices(self.latent_dim)]

        return jnp.append(grad_mu, grad_L)
