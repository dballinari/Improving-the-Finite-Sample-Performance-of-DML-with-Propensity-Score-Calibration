import numpy as np
import warnings
import sklearn.ensemble
import scipy
from enum import Enum
from sklearn.isotonic import IsotonicRegression
from venn_abers import VennAbers
from functools import partial
from .constrained_logistic import ConstrainedLogisticRegression

_MIN_T = 10**-1


def venn_abers(
    model: sklearn.ensemble, x: np.ndarray, w: np.ndarray, x_test: np.ndarray
) -> np.ndarray:
    """
    Venn ABERS calibration method from :

    Vovk, Vladimir, Ivan Petej, and Valentina Fedorova. "Large-scale probabilistic predictors
        with and without guarantees of validity."Advances in Neural Information Processing Systems 28 (2015).
        (arxiv version https://arxiv.org/pdf/1511.00213.pdf)


    Args:
        model (sklearn.ensemble): fitted propensity score model
        x (np.ndarray): features
        w (np.ndarray): label
        x_test (np.ndarray): features from test sample

    Returns:
        (np.ndarray): vector of calibrated probabilities

    """
    # Define Venn-ABERS calibrator
    va = VennAbers()

    # Fit on the training set
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        va.fit(model.predict_proba(x), w)

    # Generate probabilities and class predictions on the test set
    p_prime, _ = va.predict_proba(model.predict_proba(x_test))
    return p_prime[:, 1]


def expectation_consistent(
    model: sklearn.ensemble, x: np.ndarray, w: np.ndarray, x_test: np.ndarray
) -> np.ndarray:
    """
    Expectation consistent calibration method.


    Args:
        model (sklearn.ensemble): fitted propensity score model
        x (np.ndarray): features
        w (np.ndarray): label
        x_test (np.ndarray): features from test sample

    Returns:
        (np.ndarray): vector of calibrated probabilities

    """
    pred_class = model.predict(x)
    pred_log_prob = model.predict_log_proba(x)
    acc = np.mean(pred_class == w)

    def loss_fn(T: int = 1):
        p = np.exp(pred_log_prob / T)
        p = p / np.sum(p, axis=1)[:, np.newaxis]
        return (np.mean(np.max(p, axis=1)) - acc) ** 2

    T = scipy.optimize.minimize(loss_fn, 1, bounds=[(_MIN_T, None)]).x[0]

    pred_prob_test = np.exp(model.predict_log_proba(x_test) / T)
    pred_prob_test = pred_prob_test / np.sum(pred_prob_test, axis=1)[:, np.newaxis]

    return pred_prob_test[:, 1]


def temperature_scaling(
    model: sklearn.ensemble, x: np.ndarray, w: np.ndarray, x_test: np.ndarray
) -> np.ndarray:
    """
    Temperature scaling calibration method described in the paper "On Calibration of Modern Neural Networks" (Guo et al., 2017)
    https://proceedings.mlr.press/v70/guo17a/guo17a.pdf


    Args:
        model (sklearn.ensemble): fitted propensity score model
        x (np.ndarray): features
        w (np.ndarray): label
        x_test (np.ndarray): features from test sample

    Returns:
        (np.ndarray): vector of calibrated probabilities

    """
    pred_log_prob = model.predict_log_proba(x)

    def loss_fn(T, x, y):
        p = np.exp(x / T)
        p = p / np.sum(p, axis=1)[:, np.newaxis]
        return -(y * np.log(p[:, 1]) + (1 - y) * np.log(1 - p[:, 1])).sum()

    T = scipy.optimize.minimize(
        loss_fn,
        x0=1,
        args=(
            pred_log_prob,
            w,
        ),
        method="L-BFGS-B",
        bounds=[(_MIN_T, None)],
    ).x[0]

    pred_prob_test = np.exp(model.predict_log_proba(x_test) / T)
    pred_prob_test = pred_prob_test / np.sum(pred_prob_test, axis=1)[:, np.newaxis]

    return pred_prob_test[:, 1]


def isotonic_regression(
    model: sklearn.ensemble, x: np.ndarray, w: np.ndarray, x_test: np.ndarray
) -> np.ndarray:
    """
    Implements isotonic regression for probability calibration.

    Args:
        model (sklearn.ensemble): fitted propensity score model
        x (np.ndarray): features
        w (np.ndarray): label
        x_test (np.ndarray): features from test sample

    Returns:
        (np.ndarray): vector of calibrated probabilities
    """
    pred_prob = model.predict_proba(x)
    pred_prob_test = model.predict_proba(x_test)
    iso_reg = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    iso_reg.fit(pred_prob[:, 1], w)
    return iso_reg.transform(pred_prob_test[:, 1])


def platt_scaling(
    model: sklearn.ensemble, x: np.ndarray, w: np.ndarray, x_test: np.ndarray
) -> np.ndarray:
    """
    Implements Platt scaling procedure described in the paper "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods" (Platt, 1999)

    Args:
        model (sklearn.ensemble): fitted propensity score model
        x (np.ndarray): features
        w (np.ndarray): label
        x_test (np.ndarray): features from test sample

    Returns:
        (np.ndarray): vector of calibrated probabilities
    """
    pred_prob = model.predict_proba(x)
    pred_prob_test = model.predict_proba(x_test)
    bounds = [(0, None)]
    clf = ConstrainedLogisticRegression(include_intercept=True)
    clf.fit(pred_prob[:, 1].reshape(-1, 1), w, bounds=bounds)
    return clf.predict(pred_prob_test[:, 1].reshape(-1, 1))


def beta_scaling(
    model: sklearn.ensemble, x: np.ndarray, w: np.ndarray, x_test: np.ndarray
) -> np.ndarray:
    """
    Implements scaling procedure described in the paper "Beyond sigmoids: How to obtain well-calibrated probabilities from binary classifiers with beta calibration" (Kull, Filho and Flach, 2017)
    (https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-11/issue-2/Beyond-sigmoids--How-to-obtain-well-calibrated-probabilities-from/10.1214/17-EJS1338SI.full)

    Args:
        model (sklearn.ensemble): fitted propensity score model
        x (np.ndarray): features
        w (np.ndarray): label
        x_test (np.ndarray): features from test sample

    Returns:
        (np.ndarray): vector of calibrated probabilities
    """
    pred_prob = model.predict_proba(x)
    pred_prob_test = model.predict_proba(x_test)
    s1 = np.log(pred_prob[:, 1])
    s2 = -np.log(1 - pred_prob[:, 1])
    # join the two one dimensional arrays into a single two dimensional array
    s = np.concatenate([s1.reshape(-1, 1), s2.reshape(-1, 1)], axis=1)
    # fit the logitst regression using two inputs derived from the predicted scores
    bounds = [(0, None) for _ in range(2)]
    clf = ConstrainedLogisticRegression(include_intercept=True)
    clf.fit(s, w, bounds=bounds)
    # calculate the scaling
    s1_test = np.log(pred_prob_test[:, 1])
    s2_test = -np.log(1 - pred_prob_test[:, 1])
    s_test = np.concatenate([s1_test.reshape(-1, 1), s2_test.reshape(-1, 1)], axis=1)
    pred_prob_test_cal = clf.predict(s_test)
    # if (pred_prob_test_cal < 0.01).any():
    #     print('Error')
    return pred_prob_test_cal


class ProbabilityCalibrators(Enum):
    EXPECTATION_CONSISTENT = partial(expectation_consistent)
    TEMPERATURE_SCALING = partial(temperature_scaling)
    ISOTONIC_REGRESSION = partial(isotonic_regression)
    PLATT_SCALING = partial(platt_scaling)
    BETA_SCALING = partial(beta_scaling)
    VENN_ABERS = partial(venn_abers)
