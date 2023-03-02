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


class FastShapelet:
    def __init__(self, n_shapelets, max_shapelet_length, n_jobs=1, verbose=0):
        self.n_shapelets = n_shapelets
        self.max_shapelet_length = max_shapelet_length
        self.n_jobs = n_jobs
        self.verbose = verbose

    @staticmethod
    def _compute_collision_table(sax_strings: np.ndarray, r: int = 10) -> np.ndarray:
        """
        Computes a collision table for the given SAX strings.

        :param sax_strings: The SAX strings to compute the collision table for.
        :return: The collision table.
        """

        objs = [np.unique(multiple_sax_string) for multiple_sax_string in sax_strings]
        n_different_string = np.sum([len(obj) for obj in objs])
        idx_table = np.concatenate([[i] * len(obj) for i, obj in enumerate(objs)])

        collision_table = np.zeros((n_different_string, len(objs)), dtype=np.int32)
        objs = np.concatenate(objs, axis=0)

        random_hashes = get_random_hash(r,len(objs[0]))
        for hash_mask in random_hashes:
            projected_words = np.array([apply_mask(obj, hash_mask) for obj in objs])
            unique_words, _ = np.unique(projected_words, return_counts=True)
            for unique_word in unique_words:
                ids_to_update = np.where(unique_word == projected_words)[0]
                for id_to_update in ids_to_update:
                    collision_table[id_to_update, idx_table[projected_words == unique_word]] += 1

        return collision_table