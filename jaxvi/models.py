class LinearRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.latent_dim = 3

    def inv_T(self, ζ):
        return jnp.append(ζ[:-1], jnp.exp(ζ[-1]))

    def __call__(self, theta):
        betas = theta[:2]
        sigma = theta[2]

        beta_prior = norm.logpdf(betas, 0, 10).sum()
        sigma_prior = gamma.logpdf(sigma, a=1, scale=2).sum()
        yhat = jnp.inner(self.x, betas)
        likelihood = norm.logpdf(self.y, yhat, sigma).sum()

        return beta_prior + sigma_prior + likelihood
