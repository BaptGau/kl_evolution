import numpy as np

from kl_evolution.core.data_objects.serie import Serie
from kl_evolution.core.features.kl_divergence import KLDivergence


class ForecastEvaluation:
    @staticmethod
    def evaluate_forecast(
        train_set: Serie,
        test_set: Serie,
        forecasts: list[Serie],
        seasonal_period: int | None = None,
        normalize: bool = False,
    ) -> dict:
        """
        Evaluate a forecast against a test set based on the DKL.
        Can normalize results by the DKL of a uniform distribution to help interpretability.
        """
        min = test_set.__min__()
        max = test_set.__max__()
        mean = test_set.__avg__()
        std = test_set.__std__()
        size = test_set.__len__()

        white_noise_ref = np.random.normal(loc=mean, scale=std, size=size)
        random_walk = np.cumsum(white_noise_ref)
        uniform_ref = np.random.uniform(low=min, high=max, size=size)

        naive = [train_set.values[-1]] * size
        if seasonal_period:
            seasonal_naive = [
                train_set.values[index % seasonal_period] for index in range(size)
            ]

        returned_dict = {
            "uniform": KLDivergence.compute(p=test_set, q=Serie(values=uniform_ref)),
            "white_noise": KLDivergence.compute(
                p=test_set, q=Serie(values=white_noise_ref)
            ),
            "random_walk": KLDivergence.compute(
                p=test_set, q=Serie(values=random_walk)
            ),
            "naive": KLDivergence.compute(p=test_set, q=Serie(values=naive)),
        }

        if seasonal_period:
            returned_dict["seasonal_naive"] = KLDivergence.compute(
                p=test_set, q=Serie(values=seasonal_naive)
            )

        for forecast in forecasts:
            returned_dict[forecast.identifier] = KLDivergence.compute(
                p=test_set, q=forecast
            )

        if normalize:
            return {
                key: value / returned_dict["uniform"]
                for key, value in returned_dict.items()
            }

        return returned_dict
