import numpy as np


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