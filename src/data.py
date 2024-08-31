"""
Credit: https://github.com/uber/causalml
"""

import numpy as np
from typing import Tuple
from scipy.stats import beta

# Define constants
MIN_COVARIATES = 5


def simulate_data(mode=1, n=1000, p=5, sigma=1.0):
    """ Synthetic data in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'
    Args:
        mode (int, optional): mode of the simulation: \
            1 for difficult nuisance components and an easy treatment effect. \
            2 for a randomized trial. \
            3 for an easy propensity and a difficult baseline. \
            4 for unrelated treatment and control groups. \
            5 for a hidden confounder biasing treatment.
        n (int, optional): number of observations
        p (int optional): number of covariates (>=5)
        sigma (float): standard deviation of the error term
        adj (float): adjustment term for the distribution of propensity, e. Higher values shift the distribution to 0.
                     It does not apply to mode == 2 or 3.
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
            - ate (float): average treatment effect.
    """

    catalog = {
        1: simulate_easy_propensity_easy_baseline,
        2: simulate_difficult_propensity_easy_baseline,
        3: simulate_easy_propensity_difficult_baseline,
        4: simulate_difficult_propensity_difficult_baseline,
        5: simulate_extreme_propensity_difficult_baseline,
    }

    assert mode in catalog, "Invalid mode {}. Should be one of {}".format(
        mode, set(catalog)
    )
    assert p >= MIN_COVARIATES, "Number of covariates should be at least {}".format(
        MIN_COVARIATES
    )

    return catalog[mode](n, p, sigma)


def simulate_nuisance_and_easy_treatment(
    n=1000, p=5, sigma=1.0
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float
]:
    """Synthetic data with a difficult nuisance components and an easy treatment effect
        From Setup A in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=5)
        sigma (float): standard deviation of the error term
        adj (float): adjustment term for the distribution of propensity, e. Higher values shift the distribution to 0.
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
            - ate (float): average treatment effect.
    """

    X = np.random.uniform(size=n * p).reshape((n, -1))
    b = (
        np.sin(np.pi * X[:, 0] * X[:, 1])
        + 2 * (X[:, 2] - 0.5) ** 2
        + X[:, 3]
        + 0.5 * X[:, 4]
    )
    eta = 0.1
    e = np.maximum(
        np.repeat(eta, n),
        np.minimum(np.sin(np.pi * X[:, 0] * X[:, 1]), np.repeat(1 - eta, n)),
    )
    tau = (X[:, 0] + X[:, 1]) / 2

    w = np.random.binomial(1, e, size=n)
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    ate = np.mean(tau)  # 0.5

    return y, X, w, tau, b, e, ate


def simulate_randomized_trial(n=1000, p=5, sigma=1.0):
    """Synthetic data of a randomized trial
        From Setup B in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=5)
        sigma (float): standard deviation of the error term
        adj (float): no effect. added for consistency
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
            - ate (float): average treatment effect.
    """

    X = np.random.normal(size=n * p).reshape((n, -1))
    b = np.maximum.reduce([np.repeat(0.0, n), X[:, 0] + X[:, 1], X[:, 2]]) + np.maximum(
        np.repeat(0.0, n), X[:, 3] + X[:, 4]
    )
    e = np.repeat(0.5, n)
    tau = X[:, 0] + np.log1p(np.exp(X[:, 1]))

    w = np.random.binomial(1, e, size=n)
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    ate = np.mean(tau)

    return y, X, w, tau, b, e, ate


def simulate_easy_propensity_difficult_baseline(n=1000, p=5, sigma=1.0):
    """Synthetic data with easy propensity and a difficult baseline
        From Setup C in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=3)
        sigma (float): standard deviation of the error term
        adj (float): no effect. added for consistency
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
            - ate (float): average treatment effect.
    """

    X = np.random.uniform(size=n * p).reshape((n, -1))
    b = (
        np.sin(np.pi * X[:, 0] * X[:, 1])
        + 2 * (X[:, 2] - 0.5) ** 2
        + X[:, 3]
        + 0.5 * X[:, 4]
    )
    e = 1 / (1 + np.exp(X[:, 1] - X[:, 2]))

    tau = (X[:, 0] + X[:, 1]) / 2

    w = np.random.binomial(1, e, size=n)
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    ate = np.mean(tau)

    return y, X, w, tau, b, e, ate


def simulate_unrelated_treatment_control(n=1000, p=5, sigma=1.0):
    """Synthetic data with unrelated treatment and control groups.
        From Setup D in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=3)
        sigma (float): standard deviation of the error term
        adj (float): adjustment term for the distribution of propensity, e. Higher values shift the distribution to 0.
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
            - ate (float): average treatment effect.
    """

    X = np.random.normal(size=n * p).reshape((n, -1))
    b = (
        np.maximum(np.repeat(0.0, n), X[:, 0] + X[:, 1] + X[:, 2])
        + np.maximum(np.repeat(0.0, n), X[:, 3] + X[:, 4])
    ) / 2
    e = 1 / (1 + np.exp(-X[:, 0]) + np.exp(-X[:, 1]))

    tau = np.maximum(np.repeat(0.0, n), X[:, 0] + X[:, 1] + X[:, 2]) - np.maximum(
        np.repeat(0.0, n), X[:, 3] + X[:, 4]
    )

    w = np.random.binomial(1, e, size=n)
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    ate = np.mean(tau)

    return y, X, w, tau, b, e, ate


def simulate_difficult_propensity_easy_baseline(
    n=1000, p=5, sigma=1.0
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float
]:
    """Synthetic data with a difficult propensity component, easy baseline and
        an easy treatment effect: own creation
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=5)
        sigma (float): standard deviation of the error term
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
            - ate (float): average treatment effect.
    """

    X = np.random.uniform(size=n * p).reshape((n, -1))
    b = X[:, 0] * X[:, 1] + 2 * (X[:, 2] - 0.5) ** 2 + X[:, 3] + 0.5 * X[:, 4]

    e = 0.1 + 0.6 * beta.cdf(np.min(X[:, :2], axis=1), 2, 4)

    tau = (X[:, 0] + X[:, 1]) / 2

    w = np.random.binomial(1, e, size=n)
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    ate = np.mean(tau)

    return y, X, w, tau, b, e, ate


def simulate_easy_propensity_easy_baseline(
    n=1000, p=5, sigma=1.0
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float
]:
    """Synthetic data with a difficult propensity component, easy baseline and
        an easy treatment effect: own creation
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=5)
        sigma (float): standard deviation of the error term
        adj (float): adjustment term for the distribution of propensity, e. Higher values shift the distribution to 0.
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
            - ate (float): average treatment effect.
    """

    X = np.random.uniform(size=n * p).reshape((n, -1))
    b = X[:, 0] * X[:, 1] + 2 * (X[:, 2] - 0.5) ** 2 + X[:, 3] + 0.5 * X[:, 4]
    e = 1 / (1 + np.exp(X[:, 1] - X[:, 2]))

    tau = (X[:, 0] + X[:, 1]) / 2

    w = np.random.binomial(1, e, size=n)
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    ate = np.mean(tau)

    return y, X, w, tau, b, e, ate


def simulate_difficult_propensity_difficult_baseline(
    n=1000, p=5, sigma=1.0
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float
]:
    """Synthetic data with a difficult propensity component, easy baseline and
        an easy treatment effect: own creation
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=5)
        sigma (float): standard deviation of the error term
        adj (float): adjustment term for the distribution of propensity, e. Higher values shift the distribution to 0.
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
            - ate (float): average treatment effect.
    """

    X = np.random.uniform(size=n * p).reshape((n, -1))
    b = (
        np.sin(np.pi * X[:, 0] * X[:, 1])
        + 2 * (X[:, 2] - 0.5) ** 2
        + X[:, 3]
        + 0.5 * X[:, 4]
    )
    e = 0.1 + 0.6 * beta.cdf(np.min(X[:, :2], axis=1), 2, 4)

    tau = (X[:, 0] + X[:, 1]) / 2

    w = np.random.binomial(1, e, size=n)
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    ate = np.mean(tau)

    return y, X, w, tau, b, e, ate


def simulate_extreme_propensity_difficult_baseline(
    n=1000, p=5, sigma=1.0
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float
]:
    """Synthetic data with a difficult propensity component, easy baseline and
        an easy treatment effect: own creation
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=5)
        sigma (float): standard deviation of the error term
        adj (float): adjustment term for the distribution of propensity, e. Higher values shift the distribution to 0.
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
            - ate (float): average treatment effect.
    """

    X = np.random.uniform(size=n * p).reshape((n, -1))
    b = (
        np.sin(np.pi * X[:, 0] * X[:, 1])
        + 2 * (X[:, 2] - 0.5) ** 2
        + X[:, 3]
        + 0.5 * X[:, 4]
    )
    e = 0.05 + 0.9 * beta.cdf(np.min(X[:, :2], axis=1), 2, 4)

    tau = (X[:, 0] + X[:, 1]) / 2

    w = np.random.binomial(1, e, size=n)
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    ate = np.mean(tau)

    return y, X, w, tau, b, e, ate
