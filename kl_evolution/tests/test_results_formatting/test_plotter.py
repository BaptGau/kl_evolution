import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from kl_evolution.core.data_objects.serie import Serie
from kl_evolution.core.results_formatting.plotter import KLResultsPlotter


class TestKLResultsPlotter(unittest.TestCase):

    def setUp(self):
        self.series = MagicMock(spec=Serie)
        self.kl_results = MagicMock(spec=Serie)

        self.series.identifier = "test_series"
        self.series.index = np.arange(10)
        self.series.values = np.random.rand(10)

        self.kl_results.__avg__.return_value = 0.5
        self.kl_results.__len__.return_value = 10
        self.kl_results.values = np.random.rand(10)

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.show")
    def test_plot_kl_results_with_save_path(self, mock_show, mock_savefig):
        save_path = "test_plot.png"

        KLResultsPlotter.plot_kl_results(
            serie=self.series,
            kl_results=self.kl_results,
            save_path=save_path,
            title="Test Title",
            xlabel="Test X-axis",
            figsize=(12, 8),
        )

        mock_savefig.assert_called_once_with(save_path, bbox_inches="tight")
        mock_show.assert_not_called()

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.show")
    def test_plot_kl_results_without_save_path(self, mock_show, mock_savefig):
        KLResultsPlotter.plot_kl_results(
            serie=self.series,
            kl_results=self.kl_results,
            save_path=None,
            title=None,
            xlabel=None,
            figsize=(12, 8),
        )

        mock_savefig.assert_not_called()
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.show")
    def test_plot_kl_results_default_params(self, mock_show, mock_savefig):
        KLResultsPlotter.plot_kl_results(
            serie=self.series,
            kl_results=self.kl_results,
            save_path=None,
        )

        mock_savefig.assert_not_called()
        mock_show.assert_called_once()
