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

def _DTW(s1,s2):
    print(s1.shape, s2.shape)
    print(type(s1), type(s2))
    return  dtw.distance_fast(np.array(s1), np.array(s2), use_pruning=True)

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
    #print(type(X), type(shapelet))
    return jnp.min(
        dist_shapelet(sliding_window_view(X, len(shapelet), axis=1), shapelet),
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

dtw_jax = jax.vmap(_DTW, in_axes=(0, None))

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from abc import ABC, abstractmethod

class AbstractDTW(ABC):
    @abstractmethod
    def minimum(self, *args):
        pass

    def __call__(self, prediction, target):
        print(prediction.shape, target.shape)
        D = distance_matrix(prediction, target)
        # wlog: H >= W
        if D.shape[0] < D.shape[1]:
            D = D.T    
        H = D.shape[0]

        rows = []
        for row in range(H):
            rows.append( pad_inf(D[row], row, H-row-1) )

        model_matrix = jnp.stack(rows, axis=1)

        init = (
            pad_inf(model_matrix[0], 1, 0),
            pad_inf(model_matrix[1] + model_matrix[0, 0], 1, 0)
        )

        def scan_step(carry, current_antidiagonal):
            two_ago, one_ago = carry

            diagonal = two_ago[:-1]
            right    = one_ago[:-1]
            down     = one_ago[1:]
            best     = self.minimum(jnp.stack([diagonal, right, down], axis=-1))

            next_row = best + current_antidiagonal
            next_row = pad_inf(next_row, 1, 0)

            return (one_ago, next_row), next_row

        # Manual unrolling:
        # carry = init
        # for i, row in enumerate(model_matrix[2:]):
        #     carry, y = scan_step(carry, row)

        print(model_matrix[2:].shape)
        print(init[0].shape)

        carry, ys = jax.lax.scan(scan_step, init, model_matrix[2:], unroll=4)
        return carry[1][-1]


class DTW(AbstractDTW):
    """
    SoftDTW as proposed in the paper "Dynamic programming algorithm optimization for spoken word recognition"
    by Hiroaki Sakoe and Seibi Chiba (https://arxiv.org/abs/1703.01541)

    Expects inputs of the shape [T, D], where T is the time dimension
    and D is the feature dimension.
    """
    __name__ = 'DTW'

    def minimum(self, args):
        return jnp.min(args, axis=-1)


class SoftDTW(AbstractDTW):
    """
    SoftDTW as proposed in the paper "Soft-DTW: a Differentiable Loss Function for Time-Series"
    by Marco Cuturi and Mathieu Blondel (https://arxiv.org/abs/1703.01541)

    Expects inputs of the shape [T, D], where T is the time dimension
    and D is the feature dimension.
    """
    __name__ = 'SoftDTW'

    def __init__(self, gamma=1.0):
        assert gamma > 0, "Gamma needs to be positive."
        self.gamma = gamma
        self.__name__ = f'SoftDTW({self.gamma})'
        self.minimum_impl = self.make_softmin(gamma)

    def make_softmin(self, gamma, custom_grad=True):
        """
        We need to manually define the gradient of softmin
        to ensure (1) numerical stability and (2) prevent nans from
        propagating over valid values.
        """
        def softmin_raw(array):
            return -gamma * logsumexp(array / -gamma, axis=-1)
        
        if not custom_grad:
            return softmin_raw

        softmin = jax.custom_vjp(softmin_raw)

        def softmin_fwd(array):
            return softmin(array), (array / -gamma, )

        def softmin_bwd(res, g):
            scaled_array, = res
            grad = jnp.where(jnp.isinf(scaled_array),
                jnp.zeros(scaled_array.shape),
                jax.nn.softmax(scaled_array) * jnp.expand_dims(g, 1)
            )
            return grad,

        softmin.defvjp(softmin_fwd, softmin_bwd)
        return softmin

    def minimum(self, args):
        return self.minimum_impl(args)


# Utility functions
def distance_matrix(a, b):
    has_features = len(a.shape) > 1
    a = jnp.expand_dims(a, axis=1)
    b = jnp.expand_dims(b, axis=0)
    D = jnp.square(a - b)
    if has_features:
        D = jnp.sum(D, axis=-1)
    return D


def pad_inf(inp, before, after):
    return jnp.pad(inp, (before, after), constant_values=jnp.inf)


def DTW_distance(sliding_window_view, shapelet):

    #shapelet = jnp.expand_dims(shapelet, axis=0)
    return jnp.apply_along_axis(lambda x: DTW()(x, shapelet), -1,sliding_window_view)