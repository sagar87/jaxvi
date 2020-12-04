=====
jaxvi
=====


.. image:: https://img.shields.io/pypi/v/jaxvi.svg
        :target: https://pypi.python.org/pypi/jaxvi

.. image:: https://img.shields.io/travis/sagar87/jaxvi.svg
        :target: https://travis-ci.com/sagar87/jaxvi

.. image:: https://readthedocs.org/projects/jaxvi/badge/?version=latest
        :target: https://jaxvi.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




A naive implementation of Automatic Differentiation Variational Inference with JAX.

What is jaxvi ?
---------------

Jaxvi implements (a very naive) version of automatic differentiation 
variational inference (ADVI). It makes use of the of the JAX library which enables 
automatic differentiation and fast computations through XLA (Accelerated Linear 
Algebra). This is a private project, which is likely to contain bugs and updated 
only irregularly (for serious applications check out mature libraries such as numpyro).


Jaxvi is designed to be lightweight and contains only the most important ingredients 
to enable ADVI. The library implements mean-field and full-rank ADVI and a few optimizer 
to allow the user to train a small set of predefined models. The API is kept minimal 
and exposes only very few functions.

Quick start
-----------

In this example we perform simple ordinary linear regression. We begin by simulating data

.. code-block:: python

    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    
    np.random.seed(3)
    beta = np.array([4, 2])
    x = np.linspace(-5, 5, 1000)  
    x = np.stack([np.ones_like(x), x]).T
    y = x @ beta   
    y = stats.norm.rvs(y, 3)

to fit the regression coefficients beta as well as the standard deviation of :code:`y`, 
we use one of the predefined models of the models module.

.. code-block:: python
    
    from jaxvi.infer import ADVI, FullRankADVI
    from jaxvi.optim import Default, Adam
    from jaxvi.utils import fit
    lm = LinearRegression(x, y)

To fit the model, we simply pass the model to the :code:`fit` function, which 
by default uses an :code:`Adam` optimiser to fit the passed model. The fit function returns 
the optimised model and an array containing the ELBO loss during each step.

.. code-block:: python
    
    results, loss = fit(lm, num_steps=1000)


The fitted parameters may be inspected using the :code:`loc` and :code:`scale` method of 
the returned model.

* Free software: MIT license
* Documentation: https://jaxvi.readthedocs.io.


Features
--------

* Mean-field and full-rank ADVI

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
