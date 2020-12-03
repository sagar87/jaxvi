from abc import abstractmethod
from jaxvi.abstract import ABCMeta
from typing import Callable, NamedTuple, Tuple
import jax.numpy as jnp


class Optimizer(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self):
        pass


class DefaultState(NamedTuple):
    s: jnp.DeviceArray
    i: int


class Default(Optimizer):
    def __init__(
        self, lr: float = 0.1, tau: float = 1.0, alpha: float = 0.9, eps: float = 1e-16
    ) -> None:
        self.lr = lr
        self.tau = tau
        self.alpha = alpha
        self.eps = eps

    def __call__(self) -> Tuple[Callable, Callable]:
        def init(position: jnp.DeviceArray, gradients: jnp.DeviceArray) -> DefaultState:
            s = gradients ** 2
            return DefaultState(s, 1)

        def update(
            state: DefaultState, position: jnp.DeviceArray, gradients: jnp.DeviceArray
        ) -> Tuple[jnp.DeviceArray, DefaultState]:
            s = self.alpha * gradients ** 2 + (1 - self.alpha) * state.s
            rho = self.lr * (state.i ** (-1 / 2 + self.eps)) / (self.tau + s ** (1 / 2))
            new_position = position + rho * gradients
            return new_position, DefaultState(s, state.i + 1)

        return init, update


class AdamState(NamedTuple):
    m: jnp.DeviceArray
    v: jnp.DeviceArray
    step: int


class Adam(Optimizer):
    def __init__(
        self,
        lr: float = 0.1,
        weight_decay_rate: float = 1e-5,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-5,
    ) -> None:
        self.lr = lr
        self.weight_decay_rate = weight_decay_rate
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def __call__(self) -> Tuple[Callable, Callable]:
        def init(position: jnp.DeviceArray, gradients: jnp.DeviceArray) -> AdamState:
            m = jnp.zeros_like(position)
            v = jnp.zeros_like(position)
            return AdamState(m, v, 1)

        def update(
            state: AdamState, position: jnp.DeviceArray, gradients: jnp.DeviceArray
        ) -> Tuple[jnp.DeviceArray, AdamState]:
            m, v, step = state

            m = (1 - self.b1) * gradients + self.b1
            v = (1 - self.b2) * (gradients ** 2) + self.b2

            mhat = m / (1 - self.b1 ** (step + 1))
            vhat = v / (1 - self.b2 ** (step + 1))

            new_position = (1 - self.weight_decay_rate) * position + (
                self.lr * mhat / (jnp.sqrt(vhat) + self.eps)
            ).astype(position.dtype)

            return new_position, AdamState(m, v, step + 1)

        return init, update
