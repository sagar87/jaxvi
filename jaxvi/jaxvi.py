"""Main module."""


class ADVI(object):
    def __init__(self, model):
        """
        A native implementation of ADVI.

        Arguments:
            log_joint: function to compute the log joint p(x, theta)
            inv_T: function to map latent parameters to the real space
            latent_dim: s
        """
        self.model = model

        self.inv_T = model.inv_T
        self.jac_T = jacfwd(self.inv_T)

        # gradients
        self.grad_joint = grad(model)
        self.grad_det_J = grad(self.log_abs_det_jacobian)

        # variational parameters
        self.ϕ = jnp.zeros((2, model.latent_dim))

    def log_abs_det_jacobian(self, ζ):
        return jnp.log(jnp.abs(jnp.linalg.det(self.jac_T(ζ))))

    @property
    def loc(self):
        return self.inv_T(self.ϕ[0])

    @property
    def scale(self):
        return jnp.exp(self.ϕ[1])

    def inv_S(self, η, ϕ):
        """
        Transforms eta to zeta.
        """
        return (η * jnp.exp(ϕ[1])) + ϕ[0]

    def variational_entropy(self, ζ, ϕ):
        probs = multivariate_normal.pdf(ζ, mean=ϕ[0], cov=jnp.diag(jnp.exp(ϕ[1])))
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
        grad_ω = grad_μ * η * jnp.exp(ϕ[1]) + 1

        return jnp.stack([grad_μ, grad_ω])

    def elbo(self, η, ϕ):
        ζ = self.inv_S(η, ϕ)
        θ = self.inv_T(ζ)

        return (
            self.model(θ)
            + self.log_abs_det_jacobian(ζ)
            + self.variational_entropy(ζ, ϕ)
        )

    def step(self, i, params):
        ϕ, g, η, loss = params

        loss = index_update(loss, i, self.elbo(η[i], ϕ))
        grad_ϕ = self.grad(η[i], ϕ)

        ρ, g = get_learing_rate(grad_ϕ, g, i, 0.1)
        ϕ += ρ * grad_ϕ

        return (ϕ, g, η, loss)
