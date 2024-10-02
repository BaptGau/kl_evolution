import pandas as pd
from kl_evolution.core.data_objects.serie import Serie
from kl_evolution.core.features.ts_analyzer import ShiftedSerieAnalyzer
from kl_evolution.core.results_formatting.plotter import KLResultsPlotter

url_file = (
    "https://raw.githubusercontent.com/scikit-learn-contrib/MAPIE/master/"
    "examples/data/demand_temperature.csv"
)


if __name__ == "__main__":
    demand_df = pd.read_csv(url_file, parse_dates=True, index_col=0)
    week_period = 24 * 7
    seasonality = 24

    serie = Serie(
        values=demand_df.loc[:, "Temperature"].values,
        index=demand_df.index,
        detrend=False, # can make them vary to see the uncertainty decreasing
        deseasonalize=False, # can make them vary to see the uncertainty decreasing
        seasonal_period=seasonality,
        identifier="Temperature",
    )

    analyzer = ShiftedSerieAnalyzer(max_horizon=week_period, normalized=True)
    kl_evolution = analyzer.compute(serie=serie)

    KLResultsPlotter.plot_kl_results(serie=serie, kl_results=kl_evolution)
