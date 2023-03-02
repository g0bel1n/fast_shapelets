import numpy as np

from ._sax import SAX
from ._utils import apply_mask, get_random_hash


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
            unique_words = np.unique(projected_words)
            for unique_word in unique_words:
                ids_to_update = np.where(unique_word == projected_words)[0]
                for id_to_update in ids_to_update:
                    collision_table[id_to_update, idx_table[projected_words == unique_word]] += 1

        return collision_table
    

    @staticmethod
    def _compute_distinguishing_score(
        collision_table: np.ndarray, obj_classes: int
    ) -> np.ndarray:
        """
        Computes the distinguishing score for the given collision table.

        :param collision_table: The collision table to compute the distinguishing score for.
        :param n_samples: The number of samples.
        :return: The distinguishing score.
        """

        n_classes = np.unique(obj_classes).shape[0]

        #group collision table cols by class
        
        close2ref = np.zeros((collision_table.shape[0], n_classes))        
        for cls in np.unique(obj_classes):
            close2ref[:,int(cls)] = np.sum(collision_table[:,obj_classes == cls], axis=-1)

        farRef = np.max(close2ref) - close2ref

        return np.sum(np.abs(close2ref - farRef), axis=-1)
    
    @staticmethod
    def _find_top_k(distinguishing_scores: np.ndarray, k:int = 10):
        """
        Finds the top k distinguishing scores.

        :param distinguishing_scores: The distinguishing scores to find the top k for.
        :param k: The number of top scores to return.
        :return: The top k distinguishing scores.
        """
        return np.argsort(distinguishing_scores)[-k:]
    
    def fit(self, X, y):
        for word_len in range(1, self.max_shapelet_length + 1):
            sax_strings = SAX(dimensionnality=word_len, cardinality=4).transform(X)
            collision_table = self._compute_collision_table(sax_strings, r=10)
            distinguishing_scores = self._compute_distinguishing_score(collision_table, y)
            top_k = self._find_top_k(distinguishing_scores, k=10)
            print(top_k)
            print(distinguishing_scores[top_k])
            #### Manque la suite
       