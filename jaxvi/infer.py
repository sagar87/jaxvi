from typing import Callable, NamedTuple, Tuple
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax import jacfwd
from jax import grad

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
        self.K = model.latent_dim
        self.phi = [jnp.zeros(self.K), jnp.ones(int(self.K * (self.K + 1) / 2))]

    def L(self, ϕ):
        k = ϕ[0].shape[0]
        L = jnp.zeros((k, k))
        L = index_update(L, jnp.tril_indices(k), ϕ[1])
        return L

    def inv_L(self, ϕ):
        L = self.L(ϕ)
        return jnp.linalg.inv(L)

    def inv_S(self, η, ϕ):
        """
        Transforms eta to zeta.
        """
        return (self.L(ϕ) @ η) + ϕ[0]

    def variational_entropy(self, ζ, ϕ):
        L = self.L(ϕ)
        probs = multivariate_normal.pdf(ζ, mean=ϕ[0], cov=L @ L.T)
        return -(probs * jnp.log(probs)).sum()

    def grad(self, η, ϕ):
        """ Returns nabla mu and nabla omega """
        ζ = self.inv_S(η, ϕ)
        θ = self.inv_T(ζ)

        # compute gradients
        grad_joint = self.grad_joint(θ)
        grad_inv_t = self.jac_T(ζ)
        grad_trans = self.grad_det_J(ζ)

        grad_μ = grad_inv_t @ grad_joint + grad_trans
        # print(grad_μ, η, grad_μ * η, grad_μ * η.T, self.inv_L(ϕ).T)
        grad_L = (grad_μ * η + self.inv_L(ϕ).T)[jnp.tril_indices(ϕ[0].shape[0])]

        return [grad_μ, grad_L]

    def step(self, i, params):
        ϕ, g, η, loss = params

        loss = index_update(loss, i, self.elbo(η[i], ϕ))
        grad_ϕ = self.grad(η[i], ϕ)

        ρ_μ, g_μ = get_learing_rate(grad_ϕ[0], g[0], i, 0.01)
        ρ_L, g_L = get_learing_rate(grad_ϕ[1], g[1], i, 0.01)

        ϕ_μ = ϕ[0] + ρ_μ * grad_ϕ[0]
        ϕ_L = ϕ[1] + ρ_L * grad_ϕ[1]  # [jnp.tril_indices(ϕ[0].shape[0])]

        return ([ϕ_μ, ϕ_L], [g_μ, g_L], η, loss)

    def fit(self, num_steps: int = 1000, rng_key: int = 0):
        η = normal(PRNGKey(rng_key), shape=(num_steps, self.model.latent_dim))
        ϕ = self.ϕ
        g = [g ** 2 for g in self.grad(η[0], ϕ)]

        loss = jnp.zeros(num_steps)
        loss = index_update(loss, 0, self.elbo(η[0], ϕ))

        self.ϕ, _, _, loss = lax.fori_loop(1, num_steps, self.step, (ϕ, g, η, loss))

        return -loss
