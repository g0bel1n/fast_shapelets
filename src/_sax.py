import numpy as np
import scipy.stats as sps
from sklearn.base import TransformerMixin
from typing import Optional


def evaluate_gaussianness(X: np.ndarray, y=None) -> float:
    max_p_val = 0
    for i in range(X.shape[0]):
        _, p_val = sps.shapiro(X[i:, :])
        if p_val > max_p_val:
            max_p_val = p_val
    return max_p_val


class SAX(TransformerMixin):
    def __init__(self, dimensionnality: int = 16, cardinality: int = 4):
        self.dimensionnality = dimensionnality  # word length
        self.cardinality = cardinality  # alphabet size

    def fit(self, X: Optional[np.ndarray] = None, y=None):
        return self

    def get_splits(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        It takes a column of data, calculates the mean and standard deviation, then creates a normal
        distribution with those parameters. It then calculates the probability density function (PDF) of
        that normal distribution, and uses it to calculate the quantiles

        :param X: the feature we want to split on
        :return: The quantiles of the normal distribution.
        """

        splits = self.cardinality

        # define the normal distribution and PDF
        dist = sps.norm()
        x = np.linspace(dist.ppf(0.001), dist.ppf(0.999))
        y = dist.pdf(x)

        # calculate PPFs
        step = 1 / splits
        quantiles = np.arange(step, 1.0 - step / 2, step)
        return dist.ppf(quantiles)

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        It takes a time series and returns a string representation of it

        :param X: the data to be transformed
        :param y: the target variable
        :return: The sax representation of the data
        """
        mu = X.mean(axis=-1, keepdims=True)

        sigma = X.std(axis=-1)

        X_ = (X - mu) / sigma[:, None]

        # if evaluate_gaussianness(X_) > 0.05:
        #     raise Warning("The data is not gaussian, the SAX representation might not be accurate")

        if X.shape[1] % self.dimensionnality:
            paa_reps = [
                [
                    np.mean(el, axis=-1)
                    for el in np.array_split(X_[:, i:], self.dimensionnality, axis=1)
                ]
                for i in range(X.shape[1] % self.dimensionnality)
            ]
            paa_reps = np.array(paa_reps)
            paa_reps = np.swapaxes(paa_reps, 0, 1).T
        else:
            paa_reps = np.mean(
                np.array_split(X_, self.dimensionnality, axis=1), axis=-1
            )
            paa_reps = paa_reps.T

        print(paa_reps.shape)

        splits = self.get_splits(X_)

        raw_sax_words = np.apply_along_axis(
            lambda x: np.digitize(x, splits), -1, paa_reps
        )
        sax_words = np.apply_along_axis(
            lambda x: "".join([chr(97 + i) for i in x]), -1, raw_sax_words
        )

        return sax_words  # sax_words of shape (n_samples, n_possible_subsequences, word_length) or (n_samples, word_length) if n_possible_subsequences = 1
