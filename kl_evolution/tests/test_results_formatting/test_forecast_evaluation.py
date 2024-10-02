import unittest
from unittest.mock import MagicMock
import numpy as np
from kl_evolution.core.data_objects.serie import Serie
from kl_evolution.core.features.kl_divergence import KLDivergence
from kl_evolution.core.results_formatting.evaluation import ForecastEvaluation


class TestForecastEvaluation(unittest.TestCase):
    def setUp(self):
        self.test_set = MagicMock(spec=Serie)
        self.train_set = MagicMock(spec=Serie)

        self.test_set.__min__.return_value = 1
        self.test_set.__max__.return_value = 10
        self.test_set.__avg__.return_value = 5.5
        self.test_set.__std__.return_value = 2.0
        self.test_set.__len__.return_value = 100
        self.train_set.values = np.array([3, 4, 5, 6, 7])
        self.forecasts = [
            Serie(values=np.random.normal(5, 2, 100)),  # Mock forecast 1
            Serie(values=np.random.normal(6, 1.5, 100)),  # Mock forecast 2
        ]
        self.forecasts[0].identifier = "forecast_1"
        self.forecasts[1].identifier = "forecast_2"

    def test_evaluate_forecast_no_seasonality(self):
        KLDivergence.compute = MagicMock(side_effect=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        result = ForecastEvaluation.evaluate_forecast(
            train_set=self.train_set,
            test_set=self.test_set,
            forecasts=self.forecasts,
            seasonal_period=None,
        )

        self.assertIn("white_noise", result)
        self.assertIn("random_walk", result)
        self.assertIn("uniform", result)
        self.assertIn("naive", result)
        self.assertIn("forecast_1", result)
        self.assertIn("forecast_2", result)

        self.assertEqual(KLDivergence.compute.call_count, 6)

    def test_evaluate_forecast_with_seasonality(self):
        KLDivergence.compute = MagicMock(
            side_effect=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        )

        seasonal_period = 3

        result = ForecastEvaluation.evaluate_forecast(
            train_set=self.train_set,
            test_set=self.test_set,
            forecasts=self.forecasts,
            seasonal_period=seasonal_period,
        )

        self.assertIn("seasonal_naive", result)
        self.assertEqual(KLDivergence.compute.call_count, 7)
        self.assertAlmostEqual(result["seasonal_naive"], 0.5)

    def test_evaluate_forecast_empty_forecasts(self):
        KLDivergence.compute = MagicMock(side_effect=[0.1, 0.2, 0.3, 0.4])

        result = ForecastEvaluation.evaluate_forecast(
            train_set=self.train_set,
            test_set=self.test_set,
            forecasts=[],
            seasonal_period=None,
        )

        self.assertIn("white_noise", result)
        self.assertIn("random_walk", result)
        self.assertIn("uniform", result)
        self.assertIn("naive", result)
        self.assertEqual(len(result), 4)
        self.assertEqual(KLDivergence.compute.call_count, 4)
