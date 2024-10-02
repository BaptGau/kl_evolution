import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from kl_evolution.core.data_objects.serie import Serie
from kl_evolution.core.results_formatting.evaluation import ForecastEvaluation

url_file = (
    "https://raw.githubusercontent.com/scikit-learn-contrib/MAPIE/master/"
    "examples/data/demand_temperature.csv"
)


if __name__ == "__main__":
    demand_df = pd.read_csv(url_file, parse_dates=True, index_col=0)
    train_set = demand_df.iloc[: -24 * 4]
    test_set = demand_df.iloc[-24 * 4 :]

    sarimax = SARIMAX(
        demand_df.loc[:, "Temperature"].values,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    sarimax_fit = sarimax.fit(disp=False, method="lbfgs")

    sarimax_forecast = sarimax_fit.forecast(steps=24 * 4)

    result = ForecastEvaluation.evaluate_forecast(
        train_set=Serie.__from_pandas__(train_set, col_name="Temperature"),
        test_set=Serie.__from_pandas__(test_set, col_name="Temperature"),
        forecasts=[Serie(values=sarimax_forecast, identifier="SARIMAX")],
        seasonal_period=24,
        normalize=True,
    )

    print(result)
    pd.DataFrame.from_dict(
        data=result, orient="index", columns=["Kullback Lieber Divergences"]
    ).plot(kind="bar", title="Kullback Lieber Divergences")
    plt.show()
