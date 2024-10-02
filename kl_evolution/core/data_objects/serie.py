from dataclasses import dataclass
import numpy as np
import pandas as pd
from numpy._typing import ArrayLike


@dataclass
class Serie:
    values: ArrayLike
    index: ArrayLike | None = None
    identifier: str | None = None
    detrend: bool = False
    deseasonalize: bool = False
    seasonal_period: int | None = None

    def __post_init__(self):
        self.values = np.array(self.values, dtype=float)

        if self.deseasonalize and not self.seasonal_period:
            raise ValueError(
                "Seasonal period should be provided for deseasonalization."
            )

        if self.index is None:
            self.index = np.arange(len(self.values))
        else:
            self.index = np.array(self.index)
            if len(self.index) != len(self.values):
                raise ValueError("Index and values must have the same length.")

    def __repr__(self):
        identifier_repr = f"{self.identifier}\n" if self.identifier else ""
        return f"{identifier_repr}Index: {self.index}\n" f"Values: {self.values}"

    def __len__(self):
        return len(self.values)

    def __min__(self):
        return np.nanmin(self.values)

    def __max__(self):
        return np.nanmax(self.values)

    def __avg__(self):
        return np.nanmean(self.values)

    def __std__(self):
        return np.nanstd(self.values)

    @staticmethod
    def __from_pandas__(
        dataframe: pd.DataFrame,
        col_name: str,
        detrend: bool = False,
        deseasonalize: bool = False,
        seasonal_period: int | None = None,
    ):
        return Serie(
            values=dataframe.loc[:, col_name].values,
            index=dataframe.index,
            identifier=col_name,
            detrend=detrend,
            deseasonalize=deseasonalize,
            seasonal_period=seasonal_period,
        )

    def shift(self, shift: int):
        """
        Shift the series values by a given number of positions.
        Shifted values are filled with NaN at the beginning.
        """
        shifted_values = np.roll(self.values, shift)
        shifted_values[:shift] = np.nan
        return Serie(
            values=shifted_values, index=self.index, identifier=self.identifier
        )

    def __all_eq__(self, other_value: float) -> bool:
        """
        Check if all values in the series are equal to a given value,
        ignoring NaNs.
        """
        return np.all(np.nan_to_num(self.values) == other_value)
