import numpy as np
import scipy.stats as sps
import math 


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

def dist_shapelet(s1,s2):
    l = s1.shape[0]
    return math.sqrt(np.linalg.norm(s1-s2)/l)