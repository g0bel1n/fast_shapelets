import numpy as np
import scipy.stats as sps
from dtaidistance import dtw
import jax.numpy as jnp
import jax
from jax import jit

from functools import partial



def get_random_hash(n: int = 10, word_size: int = 10) -> np.ndarray:
    """
    > It generates a random hash of length n

    :param n: The length of the hash, defaults to 10
    :type n: int, optional
    :return: The random hash
    :rtype: np.array(dtype=np.int32)
    """
    return np.random.binomial(n=1, p=.7, size=(n, word_size)).astype(bool)

def apply_mask(word: str, mask: np.ndarray) -> str:
    """
    > It applies a mask to a word

    :param word: The word to apply the mask to
    :type word: str
    :param mask: The mask to apply to the word
    :type mask: np.array(dtype=np.int32)
    :return: The masked word
    :rtype: str
    """
    return "".join([letter for letter, mask in zip(word, mask) if mask]) #or ''.join(np.array(list(sax_string))[mask])

def evaluate_gaussianness(X: np.ndarray, y=None) -> float:
    max_p_val = 0
    for i in range(X.shape[0]):
        _, p_val = sps.shapiro(X[i:, :])
        if p_val > max_p_val:
            max_p_val = p_val
    return max_p_val

def norm_euclidean(s1,s2):
    l = s1.shape[0]
    return np.sqrt(np.linalg.norm(s1-s2)/l)

def DTW(s1,s2):
    return  dtw.distance_fast(np.array(s1,dtype=np.double), np.array(s2,dtype=np.double), use_pruning=True)

def get_splits(n_splits : int) -> np.ndarray:
    """
    It takes a column of data, calculates the mean and standard deviation, then creates a normal
    distribution with those parameters. It then calculates the probability density function (PDF) of
    that normal distribution, and uses it to calculate the quantiles

    :param X: the feature we want to split on
    :return: The quantiles of the normal distribution.
    """

    # define the normal distribution and PDF
    dist = sps.norm()
    x = np.linspace(dist.ppf(0.001), dist.ppf(0.999))
    y = dist.pdf(x)

    # calculate PPFs
    step = 1 / n_splits
    quantiles = np.arange(step, 1.0 - step / 2, step)
    return dist.ppf(quantiles)


@partial(jit, static_argnums=(1,))
def _paa_rep(X: jax.Array, dimensionnality:int) -> jax.Array:
    """
    It takes a subsequence and returns its PAA representation

    :param subseq: the subsequence to be transformed
    :return: The PAA representation of the subsequence
    """
    return jnp.array([jnp.mean(partition, axis=-1) for partition in jnp.array_split(X, dimensionnality, axis=-1)])


def _raw_sax_rep(subseq: jax.Array, splits: np.ndarray, dimensionnality: int ) -> jax.Array:
    """
    It takes a subsequence and returns its raw SAX representation

    :param subseq: the subsequence to be transformed
    :return: The raw SAX representation of the subsequence
    """
    paa_rep = _paa_rep(subseq, dimensionnality)
    return jnp.digitize(paa_rep, splits)