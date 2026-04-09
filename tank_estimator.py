import os
import numpy as np
import matplotlib.pyplot as plt


def german_tank_simulation(N, n, repetitions=10000, seed=42):
    """
    Simulácia odhadu parametra N pomocou metódy nemeckých tankov.

    Náhodne sa vyberá bez vrátenia z množiny {1, 2, ..., N}.
    Z maxima výberu X_(n) sa počíta odhad:
        N_hat = ((n + 1) / n) * X_(n) - 1
    """
    if not (1 <= n <= N):
        raise ValueError("Musí platiť 1 <= n <= N.")

    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    population = np.arange(1, N + 1)

    for i in range(repetitions):
        sample = rng.choice(population, size=n, replace=False)
        sample_max = sample.max()
        estimates[i] = ((n + 1) / n) * sample_max - 1

    return estimates


def estimator_variance(N, n):
    """
    Rozptyl odhadu podľa vzťahu:
        Var(N_hat) = ((N + 1)(N - n)) / (n(n + 2))
    """
    return (N + 1) * (N - n) / (n * (n + 2))


def plot_single_n(
    N,
    n,
    repetitions=10000,
    seed=42,
    save_plot=True,
    output_dir="grafy",
    show_plot=True
):

    estimates = german_tank_simulation(
        N=N,
        n=n,
        repetitions=repetitions,
        seed=seed
    )

    mean_estimate = estimates.mean()
    variance_estimate = estimates.var()
    theoretical_variance = estimator_variance(N, n)

    # krok medzi možnými hodnotami odhadu
    step = (n + 1) / n

    # hranice stĺpcov histogramu zarovnané na diskrétne hodnoty odhadu
    bins = np.arange(
        estimates.min() - step / 2,
        estimates.max() + 3 * step / 2,
        step
    )

    # dynamické hranice osi x podľa najväčšieho očakávaného rozptylu
    # hranice osi x podľa simulovaných hodnôt s malou rezervou
    left_margin = 1.5 * step
    right_margin = 1.0 * step

    lower_bound = max(0, estimates.min() - left_margin)
    upper_bound = estimates.max() + right_margin

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    ax.hist(
        estimates,
        bins=bins,
        edgecolor="black",
        alpha=0.85
    )

    ax.axvline(
        N,
        linestyle="--",
        linewidth=1.2,
        color="red",
        label="parameter N"
    )

    ax.axvline(
        mean_estimate,
        linestyle=":",
        linewidth=1.2,
        #color='green',
        label="stredná hodnota odhadu"
    )

    # ax.set_title(
    #     f"Rozdelenie odhadu parametra N\n"
    #     f"veľkosť výberu n = {n}\n"
    #     f"stredná hodnota odhadu = {mean_estimate:.2f}, "
    #     f"rozptyl = {variance_estimate:.2f}",
    #     fontsize=12
    # )

    ax.set_xlabel(r"odhad $\hat{N}$", fontsize=26)
    ax.set_ylabel("Počet", fontsize=26)
    ax.set_xlim(lower_bound, upper_bound)
    ax.tick_params(axis='both', labelsize=24)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=26)

    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"odhad_N_n_{n}.pdf")
        fig.savefig(file_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_separate_for_different_n(
    N,
    n_values,
    repetitions=10000,
    seed=42,
    save_plots=True,
    output_dir="grafy",
    show_plots=True
):

    for i, n in enumerate(n_values):
        plot_single_n(
            N=N,
            n=n,
            repetitions=repetitions,
            seed=seed + i,
            save_plot=save_plots,
            output_dir=output_dir,
            show_plot=show_plots
        )


# Príklad použitia
N = 100
n_values = [5, 10, 20, 40]
repetitions = 10000

plot_separate_for_different_n(
    N=N,
    n_values=n_values,
    repetitions=repetitions,
    seed=42,
    save_plots=True,
    output_dir="grafy",
    show_plots=True
)
