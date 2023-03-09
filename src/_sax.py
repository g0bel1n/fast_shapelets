from typing import Optional, Tuple

import numpy as np
import scipy.stats as sps
from sklearn.base import TransformerMixin

from ._utils import evaluate_gaussianness


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

    def _paa_rep(self, X: np.ndarray) -> np.ndarray:
        """
        It takes a subsequence and returns its PAA representation

        :param subseq: the subsequence to be transformed
        :return: The PAA representation of the subsequence
        """
        return np.array([np.mean(partition, axis=-1) for partition in np.array_split(X, self.dimensionnality, axis=-1)])

    def _raw_sax_rep(self, subseq: np.ndarray) -> np.ndarray:
        """
        It takes a subsequence and returns its raw SAX representation

        :param subseq: the subsequence to be transformed
        :return: The raw SAX representation of the subsequence
        """
        paa_rep = self._paa_rep(subseq)
        return np.digitize(paa_rep, self.splits)

    def transform(self, X: np.ndarray, y=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        It takes a time series and returns a string representation of it

        :param X: the data to be transformed
        :param y: the target variable
        :return: The sax representation of the data
        """

        # if evaluate_gaussianness(X_) > 0.05:
        #     raise Warning("The data is not gaussian, the SAX representation might not be accurate")

        self.splits = self.get_splits(X)

        raw_sax_words = np.apply_along_axis(self._raw_sax_rep, -1, X)

        sax_words = np.apply_along_axis(
            lambda x: "".join([chr(97 + i) for i in x]), -1, raw_sax_words
        )

        return sax_words  # sax_words of shape (n_samples, n_possible_subsequences) or (n_samples,) if n_possible_subsequences = 1
