import numpy as np
import scipy.stats as sps
from dtaidistance import dtw
import jax.numpy as jnp
import jax

from numpy.lib.stride_tricks import sliding_window_view



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

def DTW(s1,s2):
    return  dtw.distance_fast(np.array(s1,dtype=np.double), np.array(s2,dtype=np.double), use_pruning=True)

#DTW = jax.vmap(DTW, in_axes=(0, None))

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

def inverse_map(idx :int, id2obj_map :np.ndarray, original_2_unique_concat_array_map :np.ndarray, raw_data_subsequences :np.ndarray):
    obj_id  = np.where(id2obj_map[idx])[0][0]
    rowinrow = np.where(id2obj_map[:,obj_id])[0][0]
    subseq_idx = original_2_unique_concat_array_map[obj_id][idx-rowinrow]
    return raw_data_subsequences[obj_id, subseq_idx], obj_id, subseq_idx


def scale(X):
    mu = X.mean(axis=-1, keepdims=True)
    sigma = X.std(axis=-1)
    return (X - mu) / sigma[:, None]

def sliding_window_view_jax(
    x: jax.Array, window_shape: int, axis: int = -1
) -> jax.Array:
    """
    Creates a sliding window view of an array.

    :param x: The input array to create a sliding window view of.
    :param window_shape: The shape of the sliding window.
    :return: The sliding window view of the input array.
    """
    reshaped = sliding_window_view(np.array(x), window_shape, axis=axis)

    return jnp.array(reshaped)


def min_dist_to_shapelet(X, shapelet, dist_shapelet):
    return jnp.min(
        dist_shapelet(sliding_window_view_jax(X, len(shapelet), axis=1), shapelet),
        axis=-1,
    )


def compute_all_distances_to_shapelet(X, shapelets, dist_shapelet):
    distances_array = np.zeros((X.shape[0], shapelets.shape[0]))
    for i, shapelet in enumerate(shapelets):
        distances_array[:, i] = min_dist_to_shapelet(X, shapelet, dist_shapelet)
    return distances_array


compute_all_distances_to_shapelet_jitted = jax.jit(compute_all_distances_to_shapelet)
compute_all_distances_to_shapelet_opt = jax.vmap(
    compute_all_distances_to_shapelet, in_axes=(0, None, None)
)


def norm_euclidean(x, y):
    return jnp.sqrt(jnp.mean((x - y) ** 2, axis=-1))


norm_euclidean = jax.vmap(norm_euclidean, in_axes=(0, None))
