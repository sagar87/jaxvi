import pytest
from jaxvi.infer import ADVI


def test_proper_advi_init(dummy_model):
    advi = ADVI(dummy_model)

    assert type(advi) == ADVI
    assert advi.latent_dim == 1


def test_improper_advi_init(improper_model):

    assert ADVI(improper_model)