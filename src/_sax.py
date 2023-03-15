from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats as sps
from jax import jit
from sklearn.base import TransformerMixin

from functools import partial

from ._utils import get_splits


@partial(jit, static_argnums=1)
def _paa_rep(X: jax.Array, dimensionnality: int = 16) -> jax.Array:
    """
    It takes a subsequence and returns its PAA representation

    :param subseq: the subsequence to be transformed
    :return: The PAA representation of the subsequence
    """
    return jnp.array([jnp.mean(partition, axis=-1) for partition in jnp.array_split(X, dimensionnality, axis=-1)])


@partial(jit, static_argnums=(1,2))
def sax(X: jax.Array, dimensionnality: int = 16, word_len : int = 10) -> jax.Array:
    """
    It takes a time series and returns a string representation of it

    :param X: the data to be transformed
    :param y: the target variable
    :return: The sax representation of the data
    """

    # if evaluate_gaussianness(X_) > 0.05:
    #     raise Warning("The data is not gaussian, the SAX representation might not be accurate")

    splits = get_splits(word_len)

    @jit
    def _raw_sax_rep(subseq: jax.Array) -> jax.Array:
        """
        It takes a subsequence and returns its raw SAX representation

        :param subseq: the subsequence to be transformed
        :return: The raw SAX representation of the subsequence
        """
        paa_rep = _paa_rep(subseq, dimensionnality)
        return jnp.digitize(paa_rep, splits)
    
    return jnp.apply_along_axis(_raw_sax_rep, -1, X)