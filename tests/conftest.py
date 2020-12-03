import pytest
from jaxvi.models import Model

@pytest.fixture
def dummy_model():
    class DummyModel(Model):
        def __init__(self):
            self.latent_dim = 1
        
        def inv_T(self):
            pass

        def log_joint(self):
            pass
    
    return DummyModel()


@pytest.fixture
def improper_model():
    class DummyModel():
        def __init__(self):
            self.latent_dim = 1
        
        def inv_T(self):
            pass

        def log_joint(self):
            pass
    
    return DummyModel()