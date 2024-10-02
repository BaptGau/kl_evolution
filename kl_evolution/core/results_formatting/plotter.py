from typing import Tuple

from kl_evolution.core.data_objects.serie import Serie
import matplotlib.pyplot as plt
import mplcyberpunk as mpl

plt.style.use("cyberpunk")


class KLResultsPlotter:

    @staticmethod
    def plot_kl_results(
        serie: Serie,
        kl_results: Serie,
        save_path: str | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        figsize: Tuple[int, int] = (15, 10),
    ) -> None:

        serie_label = serie.identifier if serie.identifier else "y"
        title = title if title else f"{serie_label}'s time evolution"
        xlabel = xlabel if xlabel else "Time"

        mean_kl = kl_results.__avg__()

        fig, axes = plt.subplots(nrows=2, figsize=figsize)

        axes[0].plot(
            serie.index,
            serie.values,
            label=serie_label,
            color="C0",
        )
        axes[0].set_ylabel(serie_label)

        axes[0].set_title(title)
        axes[0].legend()

        axes[1].plot(kl_results, color="C0", label="KL evolution over lags")
        mpl.add_gradient_fill(ax=axes[1], alpha_gradientglow=0.6)
        axes[1].plot(
            [mean_kl] * kl_results.__len__(),
            color="C1",
            label="Mean KL divergence over lags",
        )

        axes[1].set_xlabel(r"$t+h$")
        axes[1].set_ylabel(r"$D_{KL}(y_t||y_{\t+h})$")
        axes[1].set_title(
            f"Kullback-Lieber evolution over lags - Avg over ref: {mean_kl:.2f}",
        )
        axes[1].legend()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()
