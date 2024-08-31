import numpy as np
import scipy
from typing import List, Tuple


class ConstrainedLogisticRegression:

    def __init__(self, include_intercept: bool = False):
        self.include_intercept = include_intercept
        self.coef = None
        self.intercept = None

    def fit(
        self, x: np.ndarray, y: np.ndarray, bounds: List[Tuple[float, float]] = None
    ):
        """
        Fit a logistic regression model with constraints on the coefficients.

        Parameters:
            x (np.ndarray): predictors with shape (number observations, number features).
            y (np.ndarray): The response variable taking values 0 and 1.
            bounds (List[Tuple[float, float]]): The bounds for the coefficients. To indicate no bound, use (None, None).

        """

        x = np.array(x)
        y = np.array(y)

        if self.include_intercept:
            x = np.column_stack((np.ones(x.shape[0]), x))

        self._p = x.shape[1]

        # add per default unconstrained intercept
        if bounds is not None and self.include_intercept and len(bounds) == self._p - 1:
            bounds = [(None, None)] + bounds

        # assert that the number of bounds is equal to the number of coefficients
        if bounds is not None:
            assert len(bounds) == self._p

        coef_fit = scipy.optimize.minimize(
            self._objective,
            x0=np.zeros(self._p),
            args=(
                x,
                y,
            ),
            jac=self._jac,
            method="L-BFGS-B",
            bounds=bounds,
        ).x

        if self.include_intercept:
            self.intercept = coef_fit[0]
            self.coef = coef_fit[1:]
        else:
            self.intercept = 0
            self.coef = coef_fit

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the response variable given the predictors and coefficients.

        Parameters:
            x (np.ndarray): predictors with shape (number observations, number features).

        Returns:
            np.ndarray: The predicted probabilities of the response variable taking value 1.

        """
        assert self.coef is not None, "Model has not been fitted yet."

        x = np.array(x)
        g = np.dot(x, self.coef) + self.intercept
        return 1 / (1 + np.exp(-g))

    @staticmethod
    def _objective(coef, x, y):
        g = np.dot(x, coef)
        loglik = np.sum(y * g - np.log(1 + np.exp(g)))
        return -loglik

    @staticmethod
    def _jac(coef, x, y):
        g = np.dot(x, coef)
        gradient = np.dot(x.T, y - 1 / (1 + np.exp(-g)))
        return -gradient
