import unittest
import numpy as np

from kl_evolution.core.data_objects.serie import Serie
from kl_evolution.core.features.ts_analyzer import ShiftedSerieAnalyzer


class TestShiftedSerieAnalyzerEndToEnd(unittest.TestCase):

    def test_kl_divergences_without_normalization(self):
        serie = Serie(values=[0.1, 0.2, 0.4, 0.2, 0.1])
        analyzer = ShiftedSerieAnalyzer(max_horizon=3, normalized=False)
        kl_divergences = analyzer.compute(serie)

        self.assertEqual(kl_divergences.__len__(), 3), "Should have 3 KL divergences"
        self.assertTrue(
            all(isinstance(kl, np.float64) for kl in kl_divergences.values)
        ), "All values should be floats"

    def test_kl_divergences_with_normalization(self):
        """
        We do not test that values are lte 1, since the random uniform distribution can be higher than the max value (i.e the informations contained about the future is worst than randomness)
        """
        serie = Serie(values=[0.1, 0.2, 0.4, 0.2, 0.1])
        analyzer = ShiftedSerieAnalyzer(max_horizon=3, normalized=True)
        kl_divergences = analyzer.compute(serie)

        self.assertEqual(kl_divergences.__len__(), 3), "Should have 3 KL divergences"
        self.assertTrue(
            all(isinstance(kl, np.float64) for kl in kl_divergences.values)
        ), "All values should be floats"

    def test_empty_serie_raise_exception(self):
        serie = Serie(values=[])
        analyzer = ShiftedSerieAnalyzer(max_horizon=3, normalized=False)
        with self.assertRaises(Exception):
            analyzer.compute(serie)
