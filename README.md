# KL Evolution

**KL Evolution** is a tool designed to evaluate the degree of randomness in time series data using **Kullback-Leibler (KL) Divergence** and to assess the performance of forecasting models. It offers two primary functionalities: analyzing the randomness of a time series over time and using KL divergence to evaluate the accuracy of forecasting models.

## Features

### 1. **Evaluate Time Series Randomness**
This feature allows you to analyze how the randomness of a time series evolves over time by using **KL divergence** on its shifted versions. By comparing a time series to its shifted counterpart, you can gain insights into the stability or volatility of its randomness.

- **Shifts**: The time series is shifted by one or more steps, and KL divergence is computed between the original series and its shifted version.
- **Randomness Measurement**: A higher KL divergence indicates greater divergence between distributions, suggesting that the time series exhibits less stability or greater randomness at those intervals.

### 2. **Evaluate Forecasting Models**
KL Evolution can be used to measure the performance of forecasting models by comparing the predicted values to actual observations over time. Using KL divergence, this method helps determine how closely the forecasted distribution aligns with the actual time series distribution.

- **Forecasting Comparison**: By computing the KL divergence between the predicted and actual time series distributions, the performance of forecasting models can be quantitatively evaluated.
- **Model Evaluation**: A lower KL divergence value implies that the model's predictions are closer to the actual outcomes, indicating better forecasting performance.

## Installation

To install the dependencies required for this project, use:
```bash
poetry install
```

## Usage
KL Evolution can be used to either measure the randomness of a time series or evaluate forecasting models. Below is a basic overview of each functionality.

### 1.1. Evaluate Time Series Randomness
To evaluate how the randomness of a time series evolves over time, you can compute the KL divergence on shifts of the time series. Example code can be found in the examples folder, demonstrating how to compute KL divergence and interpret the randomness of the time series.
```python
serie = Serie(
    values=your_serie,
    index=your_serie_index,
    identifier="Serie_name",
)

analyzer = ShiftedSerieAnalyzer(max_horizon=max_horizon_to_analyze, normalized=True)
kl_evolution = analyzer.compute(serie=serie)

KLResultsPlotter.plot_kl_results(serie=serie, kl_results=kl_evolution)
```

### 1.2. Evaluate Forecasting Models
KL Evolution also allows you to evaluate forecasting models by comparing the KL divergence between the actual and predicted distributions. Examples are provided in the examples folder to help you integrate your forecasting models and assess their performance.
```python
    result = ForecastEvaluation.evaluate_forecast(
        train_set=Serie(values=train_set, identifier="Train_set"),
        test_set=Serie(values=test_set, identifier="Test_set"),
        forecasts=[Serie(values=model_1_forecast, identifier="Model_1"),
                    Serie(values=model_2_forecast, identifier="Model_2"),
                   ...
                   ],
        seasonal_period=24,
        normalize=True,
    )
```