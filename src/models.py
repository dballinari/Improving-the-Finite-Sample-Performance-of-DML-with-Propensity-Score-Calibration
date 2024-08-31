import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone

from typing import Tuple, List
from collections import Counter

from .estimators import _under_sample_majority_treatment


def regression_model(name: str, **kwargs):
    if name == "RandomForest":
        return clone(RandomForestRegressor(n_estimators=500, **kwargs))
    elif name == "GradientBoosting":
        return clone(GradientBoostingRegressor(**kwargs))
    elif name == "Lasso":
        pipeline = Pipeline(
            [
                (
                    "polynomial_features",
                    PolynomialFeatures(degree=2, include_bias=False),
                ),
                ("scaler", StandardScaler()),
                ("regressor", clone(Lasso(**kwargs))),
            ]
        )
        return pipeline
    else:
        raise ValueError(f"Model {name} not recognized")


def classification_model(name: str, **kwargs):
    if name == "RandomForest":
        return clone(RandomForestClassifier(n_estimators=500, **kwargs))
    elif name == "GradientBoosting":
        return clone(GradientBoostingClassifier(**kwargs))
    elif name == "Lasso":
        pipeline = Pipeline(
            [
                (
                    "polynomial_features",
                    PolynomialFeatures(degree=2, include_bias=False),
                ),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    clone(
                        LogisticRegression(penalty="l1", solver="liblinear", **kwargs)
                    ),
                ),
            ]
        )
        return pipeline
    else:
        raise ValueError(f"Model {name} not recognized")


def hyperparameters_grid(name: str, classification: bool = False):
    if name == "RandomForest":
        return {
            "max_depth": [1, 2, 3, 5, 10, 20],
            "min_samples_leaf": [5, 10, 15, 20, 30, 50],
        }
    elif name == "GradientBoosting":
        return {
            "n_estimators": [5, 10, 25, 50, 100, 200, 500],
            "learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
            "max_depth": [1, 2, 3, 5, 10],
        }
    elif name == "Lasso":
        if classification:
            return {"classifier__C": [0.005, 0.01, 0.05, 0.1, 0.5, 0.8, 1]}
        else:
            return {"regressor__alpha": [0.005, 0.01, 0.05, 0.1, 0.5, 0.8, 1]}
    else:
        raise ValueError(f"Model {name} not recognized")


def tune_nuisances(
    y: np.ndarray,
    w: np.ndarray,
    x: np.ndarray,
    model_name: dict,
    nfolds: int = 2,
    n_jobs_cv: int = -1,
    under_sample: bool = False,
    **kwargs,
) -> Tuple[dict, dict, dict]:
    if under_sample:
        y, w, x = _under_sample_majority_treatment(y, w, x)
    # find optimal hyperparameters
    classification = False
    param_grid = hyperparameters_grid(model_name, classification)
    hyperparameters_reg_treated = _cv(
        x[w == 1],
        y[w == 1],
        nfolds,
        regression_model(name=model_name, **kwargs),
        param_grid,
        n_jobs_cv,
    )
    hyperparameters_reg_not_treated = _cv(
        x[w == 0],
        y[w == 0],
        nfolds,
        regression_model(name=model_name, **kwargs),
        param_grid,
        n_jobs_cv,
    )
    classification = True
    param_grid = hyperparameters_grid(model_name, classification)
    hyperparameters_propensity = _cv(
        x,
        w,
        nfolds,
        classification_model(name=model_name, **kwargs),
        param_grid,
        n_jobs_cv,
    )
    return (
        hyperparameters_reg_treated,
        hyperparameters_reg_not_treated,
        hyperparameters_propensity,
    )


def _cv(
    x: np.ndarray,
    y: np.ndarray,
    nfolds: int,
    estimator,
    param_grid: dict,
    n_jobs: int = -1,
) -> Tuple[int, int]:
    grid = GridSearchCV(estimator, param_grid, cv=nfolds, n_jobs=n_jobs)
    grid.fit(x, y)
    best = grid.best_params_
    return {k.split("__")[-1]: v for k, v in best.items()}


def modes_of_values(list_of_dicts: List[dict]) -> dict:
    """
    Given a list of dictionaries, returns the most common value for each key
    """
    keys = list(set().union(*list_of_dicts))
    return {
        key: Counter([d[key] for d in list_of_dicts]).most_common(1)[0][0]
        for key in keys
    }
