import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ._sax import SAX
from ._utils import apply_mask, get_random_hash,dist_shapelet
from ._split import Split

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
    
    @staticmethod
    def _define_splits(splits,shapelet):
        splits_objects = []
        for split in splits :
            splits_objects.append(Split(split,shapelet))
        return splits_objects
    

    @staticmethod
    def _format_raw_seq(raw_data_sequences,X,dim):
        split1, split2 = np.array_split(np.hstack(raw_data_sequences), X.shape[1] % dim, axis=1)
        raw = []
        for i in range(X.shape[0]):
            raw.append(split1[i,:])
            raw.append(split2[i,:])
        return raw

    def fit(self, X, y):
        for word_len in range(1, self.max_shapelet_length + 1):
            cardinality =4
            sax_strings, raw_data_subsequences = SAX(dimensionnality=word_len, cardinality=cardinality).transform(X)
            collision_table = self._compute_collision_table(sax_strings, r=10)
            distinguishing_scores = self._compute_distinguishing_score(collision_table, y)
            top_k = self._find_top_k(distinguishing_scores, k=10)
            print(top_k)
            print(distinguishing_scores[top_k])
            print(raw_data_subsequences)
            #raw_data_subsequences = self._format_raw_seq(raw_data_subsequences,X,word_len)
            tscand = raw_data_subsequences.reshape(-1, raw_data_subsequences.shape[-1])[top_k] # [raw_data_subsequences[k] for k in top_k]
            print(tscand)
            print(tscand.shape)

            mu = X.mean(axis=-1, keepdims=True)
            sigma = X.std(axis=-1)
            X_ = (X - mu) / sigma[:, None]

            getallsubseq = sliding_window_view(X_,word_len,axis=1)

            distances = [np.apply_along_axis(
                            lambda x: dist_shapelet(x , cand), -1, getallsubseq) 
                        for cand in tscand] 

            min_dist = np.apply_along_axis(np.min, -1, distances)

            max_gain , min_gap = np.inf, 0
            for k,dlist in enumerate(min_dist):
                splits = self._define_splits([np.where(dlist > d ,1,0) for d in dlist],tscand[k])
                info_gain_splits = [split.info_gain(y) for split in splits]
                gaps = [split.gap(y) for split in splits]
                max_gain_cand = np.where(info_gain_splits == np.max(info_gain_splits))[0]
                if max_gain_cand.shape[0] > 1:
                    max_gap = np.argmax([gaps[m] for m in max_gain_cand])
                    best_split = splits[max_gap]
                else:
                    best_split = splits[max_gain_cand[0]]
                if (best_split.gain > max_gain) or (best_split.gain == max_gain and best_split.gap > min_gap):
                    max_gain = best_split.gain
                    min_gap = best_split.gap
                    shapelet = best_split.shapelet
        return shapelet

       