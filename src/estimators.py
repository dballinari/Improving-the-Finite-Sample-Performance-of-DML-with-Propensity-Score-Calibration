import numpy as np
from typing import Tuple
import sklearn.ensemble
from sklearn.base import clone
import scipy
from .calibration import ProbabilityCalibrators


def estimate_ate(
    y: np.ndarray,
    w: np.ndarray,
    x: np.ndarray,
    model_reg_treated: sklearn.ensemble,
    model_reg_not_treated: sklearn.ensemble,
    model_propensity: sklearn.ensemble,
    e: np.ndarray = None,
    nfolds: int = 2,
) -> Tuple[np.ndarray, list, list]:

    if e is not None:
        # initialize data for cross-fitting
        y, w, x, e, idx = _init_estimation(y, w, x, e, nfolds)

    else:
        # initialize data for cross-fitting
        y, w, x, idx = _init_estimation(y, w, x, nfolds=nfolds)

    # initialize arrays for predictions with nan
    y_pred_treated = _nan_array(y.shape)
    y_pred_not_treated = _nan_array(y.shape)
    w_pred = _nan_array((y.shape[0], len(ProbabilityCalibrators) + 1))

    # loop over folds
    for i in range(nfolds):
        (y_train, w_train, x_train), (y_test, w_test, x_test), idx_test = _get_folds(
            y, w, x, idx, i
        )
        # if train and/or test sample have no treated or no non-treated, set tau to nan
        if (
            (np.sum(w_train == 1) == 0)
            or (np.sum(w_train == 0) == 0)
            or (np.sum(w_test == 1) == 0)
            or (np.sum(w_test == 0) == 0)
        ):
            continue

        # predict outcomes using data on the treated
        y_pred_treated[idx_test] = _regression_prediction(
            clone(model_reg_treated),
            x_train[w_train == 1, :],
            y_train[w_train == 1],
            x_test,
        )

        # predict outcomes using data on the non-treated
        y_pred_not_treated[idx_test] = _regression_prediction(
            clone(model_reg_not_treated),
            x_train[w_train == 0, :],
            y_train[w_train == 0],
            x_test,
        )

        # predict treatment probabilities
        w_pred[idx_test, :] = _classification_prediction(
            clone(model_propensity), x_train, w_train, x_test, w_test, calibration=True
        )

    tau = _compute_tau(y, w, y_pred_treated, y_pred_not_treated, w_pred)

    tau_reweighted = _compute_tau(
        y, w, y_pred_treated, y_pred_not_treated, w_pred[:, 0], reweight=True
    )
    # combine all taus in one array: DML, ...len(PortfolioCalibrators)..., reweighted DML
    tau = np.concatenate((tau, tau_reweighted), axis=1)
    estimate_ate, variance_ate = _get_mean_and_variance(tau)

    # compute learner metrics
    y_error = y - y_pred_treated * w - y_pred_not_treated * (1 - w)
    mse = np.mean(y_error**2)
    mse = np.array([mse] * tau.shape[1])
    mse_treated = np.mean(y_error[w == 1] ** 2)
    mse_treated = np.array([mse_treated] * tau.shape[1])
    mse_not_treated = np.mean(y_error[w == 0] ** 2)
    mse_not_treated = np.array([mse_not_treated] * tau.shape[1])
    var_y = np.var(y)
    var_y = np.array([var_y] * tau.shape[1])
    brier = np.mean((w[:, np.newaxis] - w_pred) ** 2, axis=0)
    brier = np.concatenate((brier, brier[0:1]), axis=0)
    var_w = np.var(w)
    var_w = np.array([var_w] * tau.shape[1])

    # approach names
    approach_names = (
        ["DML"]
        + [calibrator.name for calibrator in ProbabilityCalibrators]
        + ["REWEIGHT_DML"]
    )

    # metrics is a matrix with rows corresponding to the different approaches and columns to the different metrics
    results = np.stack(
        (
            estimate_ate,
            variance_ate,
            mse,
            mse_treated,
            mse_not_treated,
            var_y,
            brier,
            var_w,
        ),
        axis=1,
    )

    # name of the metrics
    evaluation_names = [
        "ate",
        "ate_variance",
        "mse",
        "mse_treated",
        "mse_not_treated",
        "var_y",
        "brier",
        "var_w",
    ]

    # if the true propensity score is known, extend the metrics to include MSE, bias and variance of the estimated propensity score
    if e is not None:
        mse_e = np.mean((e[:, np.newaxis] - w_pred) ** 2, axis=0)
        mse_e = np.concatenate((mse_e, mse_e[0:1]), axis=0)
        bias_e = np.mean((e[:, np.newaxis] - w_pred), axis=0)
        bias_e = np.concatenate((bias_e, bias_e[0:1]), axis=0)
        var_e = np.var(w_pred)
        var_e = np.array([var_e] * tau.shape[1])

        results_e = np.stack((mse_e, bias_e, var_e), axis=1)

        results = np.concatenate((results, results_e), axis=1)

        evaluation_names.extend(["mse_e", "bias_e", "var_e"])

    return results, approach_names, evaluation_names


def _get_mean_and_variance(x: np.ndarray) -> Tuple[float, float]:
    return (np.nanmean(x, axis=0), np.nanvar(x, axis=0) / np.sum(~np.isnan(x), axis=0))


def _regression_prediction(
    model: sklearn.ensemble,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    # fit model
    model.fit(x_train, y_train)
    # predict outcomes
    y_pred = model.predict(x_test)
    return y_pred


def _classification_prediction(
    model: sklearn.ensemble,
    x_train: np.ndarray,
    w_train: np.ndarray,
    x_test: np.ndarray,
    w_test: np.ndarray,
    calibration: bool = False,
) -> np.ndarray:
    # fit classification model
    model.fit(x_train, w_train)
    # predict treatment probabilities
    w_pred = model.predict_proba(x_test)[:, 1]
    if calibration:
        w_pred_cal = _calibrate_propensity(model, x_test, w_test)
        w_pred = np.concatenate((_add_second_axis(w_pred), w_pred_cal), axis=1)
    return w_pred


def _calibrate_propensity(
    model: sklearn.ensemble, x: np.ndarray, w: np.ndarray, nfolds: int = 2
):

    n = x.shape[0]
    idx = np.random.choice(np.arange(n), size=n, replace=False)
    idx = np.array_split(idx, nfolds)
    calibrated_propensity = np.zeros((n, len(ProbabilityCalibrators)))
    # split data in two random samples
    for i in range(nfolds):
        idx_valid = idx[i]
        idx_test = np.concatenate(idx[:i] + idx[(i + 1) :])
        for j, calibrator in enumerate(ProbabilityCalibrators):
            calibrated_propensity[idx_test, j] = calibrator.value(
                model, x[idx_valid, :], w[idx_valid], x[idx_test, :]
            )

    return calibrated_propensity


def _init_estimation(
    y: np.ndarray,
    w: np.ndarray,
    x: np.ndarray,
    e: np.ndarray = None,
    nfolds: int = 2,
    under_sample: bool = False,
) -> np.ndarray:
    if under_sample:
        y, w, x, e = _under_sample_majority_treatment(y, w, x, e)
    n = x.shape[0]
    idx = np.random.choice(np.arange(n), size=n, replace=False)
    idx = np.array_split(idx, nfolds)

    if e is not None:
        return y, w, x, e, idx
    else:
        return y, w, x, idx


def _get_folds(
    y: np.ndarray, w: np.ndarray, x: np.ndarray, fold_indices: np.array, fold_idx: int
):
    # split sample into train and test
    idx_test = fold_indices[fold_idx]
    idx_train = np.concatenate(fold_indices[:fold_idx] + fold_indices[(fold_idx + 1) :])
    x_train = x[idx_train, :]
    y_train = y[idx_train]
    w_train = w[idx_train]
    x_test = x[idx_test, :]
    y_test = y[idx_test]
    w_test = w[idx_test]
    return (y_train, w_train, x_train), (y_test, w_test, x_test), idx_test


def _nan_array(shape: Tuple[int, int]) -> np.ndarray:
    nan_array = np.full(shape, np.nan, dtype=float)
    return nan_array


def _add_second_axis(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x[:, np.newaxis]
    return x


def _compute_tau(
    y: np.array,
    w: np.array,
    y_pred_treated: np.array,
    y_pred_not_treated: np.array,
    w_pred: np.array,
    reweight: bool = False,
    tol_propensity: float = 1e-10,
):
    y = _add_second_axis(y)
    w = _add_second_axis(w)
    y_pred_treated = _add_second_axis(y_pred_treated)
    y_pred_not_treated = _add_second_axis(y_pred_not_treated)
    w_pred = _add_second_axis(w_pred)

    # winsorize propensity scores
    w_pred = np.minimum(np.maximum(w_pred, tol_propensity), 1 - tol_propensity)

    # define weights
    weight_treated = w / w_pred
    weight_not_treated = (1 - w) / (1 - w_pred)

    # normalize weights
    if reweight:
        weight_treated = weight_treated / np.nanmean(weight_treated, axis=0)
        weight_not_treated = weight_not_treated / np.nanmean(weight_not_treated, axis=0)

    tau = (
        y_pred_treated
        - y_pred_not_treated
        + weight_treated * (y - y_pred_treated)
        - weight_not_treated * (y - y_pred_not_treated)
    )
    return tau


def _under_sample_majority_treatment(
    y: np.ndarray, w: np.ndarray, x: np.ndarray, e: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = x.shape[0]
    # under-sample the majority class
    n_treated = np.sum(w)
    n_not_treated = n - n_treated
    if n_treated > n_not_treated:
        # under-sample treated
        idx = np.where(w == 1)[0]
        idx = np.random.choice(idx, size=n_not_treated, replace=False)
        idx = np.concatenate((idx, np.where(w == 0)[0]))
    else:
        # under-sample not treated
        idx = np.where(w == 0)[0]
        idx = np.random.choice(idx, size=n_treated, replace=False)
        idx = np.concatenate((idx, np.where(w == 1)[0]))
    x = x[idx, :]
    y = y[idx]
    w = w[idx]
    if e is not None:
        e = w[idx]
        return y, w, x, e
    else:
        return y, w, x
