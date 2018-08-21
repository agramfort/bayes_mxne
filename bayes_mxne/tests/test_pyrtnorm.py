# Author : Alex Gramfort, <alexandre.gramfort@inria.fr>

import numpy as np
from numpy.testing import assert_array_equal
from bayes_mxne.pyrtnorm import rtnorm


def test_rtnorm_random_state():
    """Test that rtnorm handles the numpy random state properly"""
    a, b = -2., 2.
    rng = np.random.RandomState(42)
    r1 = rtnorm(a, b, mu=0., sigma=1., size=10000, random_state=rng)

    r2 = rtnorm(a, b, mu=0., sigma=1., size=10000, random_state=rng)
    assert np.all(r1 != r2)

    rng = np.random.RandomState(42)
    r3 = rtnorm(a, b, mu=0., sigma=1., size=10000, random_state=rng)
    assert_array_equal(r1, r3)
