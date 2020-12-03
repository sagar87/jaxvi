from jax.random import PRNGKey, normal
import jax.numpy as jnp
from jax import jit, lax
from jax.ops import index_update
from jaxvi.infer import ADVI
from jaxvi.models import Model
from jaxvi.optim import Optimizer, Adam


def fit(
    model: Model,
    optim: Optimizer = Adam,
    infer=ADVI,
    num_steps: int = 1000,
    rng_key: int = 0,
):
    eta = normal(PRNGKey(rng_key), shape=(num_steps, model.latent_dim))
    advi = infer(model)
    phi = advi.phi
    grad_phi = advi.grad(eta[0], phi)

    #
    loss = jnp.zeros(num_steps)
    loss = index_update(loss, 0, advi.elbo(eta[0], phi))

    # intialise optimiser
    init, update = optim()
    state = init(phi, grad_phi)

    @jit
    def step(i, params):
        phi, state, loss, eta = params
        loss = index_update(loss, i, advi.elbo(eta[i], phi))
        grad = advi.grad(eta[i], phi)
        new_phi, new_state = update(state, phi, grad)
        return new_phi, new_state, loss, eta

    phi, state, loss, _ = lax.fori_loop(1, num_steps, step, (phi, state, loss, eta))

    advi.phi = phi

    return advi, -loss
