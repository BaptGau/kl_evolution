import copy

import numpy as np
from kl_evolution.core.data_objects.serie import Serie
from kl_evolution.core.features.kl_divergence import KLDivergence


class ShiftedSerieAnalyzer:
    """
    Class used to compute the Kullback-Leibler divergence between a serie and a shifted version of itself.
    """

    def __init__(
        self,
        max_horizon: int,
        normalized: bool = True,
    ):
        """
        :param max_horizon: The maximum horizon of the shift
        :param normalized: Whether to normalize the Kullback-Leibler divergence by the KL on a uniform distribution
        """
        self.max_horizon = max_horizon
        self.normalized = normalized

    @staticmethod
    def check_if_serie_is_not_empty(serie: Serie) -> bool:
        return serie.__len__() > 1

    def compute(self, serie: Serie) -> Serie:
        """
        Compute the Kullback-Leibler divergences between a serie and shifted versions of itself.
        If the serie has been detrended or deseasonalized, add a constant to the values before the computation to avoid NaN results.
        We therefore recommand you to normalize the results for the scale not to be wrong.

        :param serie: The series to analyze.
        :return: The Kullback-Leibler divergences between the series and shifted versions of itself.
        """
        if not self.__is_valid_serie(serie):
            raise ValueError("The values should be non-empty")

        modified_serie = self.__is_modified(serie)
        original_values = None

        if modified_serie:
            original_values = self.__modify_serie_values(serie)

        kl_divergences = self.__compute_kl_for_shifts(serie)

        if self.normalized:
            kl_divergences = self.__normalize_kl(serie, kl_divergences)

        if modified_serie:
            serie.values = original_values
            del original_values

        return Serie(values=kl_divergences, identifier="KL divergences over shifts")

    def __is_valid_serie(self, serie: Serie) -> bool:
        """Check if the series is valid (not empty)."""
        return self.check_if_serie_is_not_empty(serie=serie)

    def __is_modified(self, serie: Serie) -> bool:
        """Check if the series is either deseasonalized or detrended."""
        return serie.deseasonalize or serie.detrend

    def __modify_serie_values(self, serie: Serie) -> np.ndarray:
        """
        Modify the series values by adding a constant to avoid NaN results, if deseasonalized/detrended.
        :param serie: The series object to modify.
        :return: The original values before modification.
        """
        original_values = copy.deepcopy(serie.values)
        serie.values = serie.values + 1000
        return original_values

    def __compute_kl_for_shifts(self, serie: Serie) -> list:
        """
        Compute KL divergences for each shift up to the maximum horizon.
        :param serie: The original series.
        :return: A list of KL divergence values.
        """
        kl_divergences = []
        for shift in range(1, self.max_horizon + 1):
            shifted_serie = Serie(values=serie.values).shift(shift=shift)
            kl_divergences.append(KLDivergence.compute(serie, shifted_serie))
        return kl_divergences

    def __normalize_kl(self, serie: Serie, kl_divergences: list) -> list:
        """
        Normalize KL divergences by dividing by a reference KL divergence.
        :param serie: The original series.
        :param kl_divergences: List of computed KL divergences.
        :return: List of normalized KL divergences.
        """
        ref_kl = KLDivergence.compute(
            p=serie,
            q=Serie(
                values=np.random.uniform(
                    low=serie.__min__(), high=serie.__max__(), size=serie.__len__()
                ),
            ),
        )
        return [kl / ref_kl for kl in kl_divergences]
