import numpy as np

from kl_evolution.core.data_objects.serie import Serie
from kl_evolution.core.features.kl_divergence import KLDivergence


class TestDKL:
    def setup_method(self):
        pass

    def test_it_should_return_zero_when_p_is_q(self):
        p = Serie(values=[1, 2, 3])
        q = Serie(values=[1, 2, 3])
        dkl = KLDivergence.compute(p=p, q=q)

        assert dkl == 0.0, "DKL should be 0 when p is equal to q"

    def test_it_should_return_non_zero_when_p_is_not_q(self):
        p = Serie(values=[1, 2, 3])
        q = Serie(values=[3, 2, 1])
        dkl = KLDivergence.compute(p=p, q=q)
        assert dkl != 0.0, "DKL should be non-zero when p is not equal to q"

    def test_it_should_return_zero_when_p_is_zero(self):
        p = Serie(values=[0, 0, 0])
        q = Serie(values=[1, 2, 3])
        dkl = KLDivergence.compute(p=p, q=q)
        assert dkl == 0.0, "DKL should be 0 when p is zero"

    def test_it_should_return_zero_when_both_are_zeros(self):
        p = Serie(values=[0, 0, 0])
        q = Serie(values=[0, 0, 0])
        dkl = KLDivergence.compute(p=p, q=q)
        assert dkl == 0.0, "DKL should be 0 when both are zeros"

    def test_it_should_return_inf_if_q_is_zero(self):
        p = Serie(values=[1, 2, 3])
        q = Serie(values=[0, 0, 0])
        dkl = KLDivergence.compute(p=p, q=q)
        assert dkl == float("inf"), "DKL should be inf if q is zero"

    def test_it_should_be_positive(self):
        p = Serie(values=np.random.randn(100))
        q = Serie(values=np.random.randn(100))
        dkl = KLDivergence.compute(p=p, q=q)
        assert dkl > 0, "DKL should be positive"

    def test_it_should_omit_nan_values(self):
        p = Serie(values=[np.nan, 1, 2, 3])
        q = Serie(values=[1, np.nan, 3, 4])
        dkl = KLDivergence.compute(p=p, q=q)
        assert dkl > 0, "DKL should be positive and omit nan values"
