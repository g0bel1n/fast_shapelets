import jax
import jax.numpy as jnp
import numpy as np

from numpy.lib.stride_tricks import sliding_window_view

from tqdm import tqdm

from ._sax import sax
from ._split import Split
from ._utils import DTW, get_random_hash, inverse_map, norm_euclidean, scale


class Shapelet:
    def __init__(
        self,
        value,
        initial_sample_id=None,
        start_initial_sample=None,
        gain=None,
        gap=None,
    ):
        self.value = value
        self.gain = gain
        self.gap = gap
        self.initial_sample = initial_sample_id
        self.start_initial_sample = start_initial_sample


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


class FastShapelets:
    def __init__(
        self,
        max_shapelet_length: int,
        min_shapelet_length: int = 100,
        n_jobs=1,
        verbose=0,
        cardinality: int = 10,
        r: int = 10,
        dimensionality: int = 16,
    ):
        self.max_shapelet_length = max_shapelet_length
        self.min_shapelet_length = min_shapelet_length
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.cardinality = cardinality
        self.r = r
        self.dimensionality = dimensionality

    @staticmethod
    def _compute_collision_table(sax_strings: np.ndarray, r: int = 10) -> np.ndarray:
        """
        Computes a collision table for the given SAX strings.

        :param sax_strings: The SAX strings to compute the collision table for.
        :return: The collision table.
        """

        objs = []
        map_idxs = []
        for multiple_sax_string in sax_strings:
            unique_strings, idxs = jnp.unique(
                multiple_sax_string, axis=0, return_index=True
            )

            objs.append(unique_strings)
            map_idxs.append(idxs)

            # = [(str1_obj1, str2_obj1), (str1_obj2, str2_obj2), ...]
        n_different_string = sum(obj.shape[0] for obj in objs)
        # idx_table = jnp.concatenate([jnp.array([i] * len(obj)) for i, obj in enumerate(objs)]) # = [0, 0, 1, 1, ...]
        # idx_table = np.zeros_like(collision_table)

        collision_table = np.zeros((n_different_string, len(objs)), dtype=np.int32)
        # idx_table = jnp.concatenate([jnp.array([[0]*max(0, i-1) + [1] + [0]*(len(objs)-i)] * len(obj)) for i, obj in enumerate(objs)], axis=0)
        idx_table = np.zeros_like(collision_table)
        idx = 0

        for i, obj in enumerate(objs):
            idx_table[idx : idx + len(obj), i] = 1
            idx += len(obj)

        idx_table = jnp.array(idx_table)

        objs = jnp.concatenate(objs, axis=0)

        random_hashes = jnp.array(get_random_hash(objs.shape[1], r)).T

        print("here")
        for hash_mask in random_hashes:
            projected_words = objs[:, hash_mask]
            u, indices = jnp.unique(projected_words, axis=0, return_inverse=True)
            # interm = np.zeros((u.shape[0], len(objs)))
            # interm[indices, np.arange(len(indices))] = 1
            # #interm = jnp.array(interm)
            # c = interm@idx_table
            c = (
                jnp.zeros((u.shape[0], len(objs)))
                .at[indices, np.arange(len(indices))]
                .set(1)
                @ idx_table
            )
            for i in range(c.shape[0]):
                bool_mask = indices == i
                collision_table[bool_mask, :] += c[i]

        return collision_table, map_idxs, idx_table

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

        # group collision table cols by class

        close2ref = np.zeros((collision_table.shape[0], n_classes))
        for cls in np.unique(obj_classes):
            close2ref[:, int(cls)] = np.sum(
                collision_table[:, obj_classes == cls], axis=-1
            )

        farRef = np.max(close2ref) - close2ref

        return np.sum(np.abs(close2ref - farRef), axis=-1)

    @staticmethod
    def _find_top_k(distinguishing_scores: np.ndarray, k: int = 10):
        """
        Finds the top k distinguishing scores.

        :param distinguishing_scores: The distinguishing scores to find the top k for.
        :param k: The number of top scores to return.
        :return: The top k distinguishing scores.
        """
        return np.argsort(distinguishing_scores)[-k:]

    @staticmethod
    def _define_splits(splits, shapelet):
        return [Split(split, shapelet) for split in splits]

    @staticmethod
    def _format_raw_seq(raw_data_sequences, X, dim):
        split1, split2 = np.array_split(
            np.hstack(raw_data_sequences), X.shape[1] % dim, axis=1
        )
        raw = []
        for i in range(X.shape[0]):
            raw.extend((split1[i, :], split2[i, :]))
        return raw

    def fit(self, X, y, dist_shapelet=norm_euclidean):
        shapelets = {}

        X_ = scale(X)

        dist_shapelet = jax.jit(dist_shapelet)
        dist_shapelet = jax.vmap(dist_shapelet, in_axes=(0, None))

        for _len in range(self.min_shapelet_length, self.max_shapelet_length + 1):
            raw_data_subsequences, tscand = self.get_candidates_shapelets(y, X_, _len)
            print("computed candidates shapelets")

            min_dist = compute_all_distances_to_shapelet(
                X_, jnp.array([a.value for a in tscand]), dist_shapelet
            )
            print("computed distances")
            # min_dist.shape = (n_samples, n_shapelets) #minimum distance of each shapelet to each sample
            shapelets[_len] = self.get_best_shapelet(y, tscand, min_dist)
            print("computed best shapelets")

        self.shapelets = shapelets

    def get_best_shapelet(self, y, tscand, min_dist):
        max_gain, min_gap = np.inf, 0
        shapelet = None

        assert (
            min_dist.shape[1] == tscand.shape[0]
        ), "min_dist.shape[0] != tscand.shape[0]"
        for k, dlist in enumerate(min_dist.T):
            splits = self._define_splits(
                [(dlist > d).astype(int) for d in dlist], tscand[k]
            )
            info_gain_splits = [split.info_gain(y) for split in splits]
            gaps = [split.gap(y) for split in splits]

            ### get the best split for a given shapelet

            max_gain_cand = np.argmax(info_gain_splits, keepdims=True)
            if max_gain_cand.shape[0] > 1:
                max_gap = np.argmax([gaps[m] for m in max_gain_cand])
                best_split = splits[max_gap]

            else:
                best_split = splits[max_gain_cand[0]]

                # compare the current best split with the previous best split to find the best shapelet
            if (best_split.gain < max_gain) or (
                best_split.gain == max_gain and best_split.gap > min_gap
            ):
                max_gain = best_split.gain
                min_gap = best_split.gap
                shapelet = best_split.shapelet

        shapelet.gain = max_gain
        shapelet.gap = min_gap

        return shapelet

    def get_candidates_shapelets(self, y, X_, _len):
        raw_data_subsequences = sliding_window_view(X_, _len, axis=1)
        sax_strings = sax(
            raw_data_subsequences,
            cardinality=self.cardinality,
            dimensionality=self.dimensionality,
        )
        collision_table, map_idxs, idx_table = self._compute_collision_table(
            sax_strings, r=self.r
        )
        distinguishing_scores = self._compute_distinguishing_score(collision_table, y)
        top_k = self._find_top_k(distinguishing_scores, k=10)

        tscand = np.array(
            [
                Shapelet(
                    *inverse_map(
                        _id,
                        idx_table=idx_table,
                        map_idxs=map_idxs,
                        raw_data_subsequences=raw_data_subsequences,
                    )
                )
                for _id in top_k
            ]
        )
        return raw_data_subsequences, tscand

    def get_shapelets(self):
        return self.shapelets

    def transform(self, X, dist_shapelet):
        shapelets = np.array([el.value for el in self.get_shapelets()])
        return compute_all_distances_to_shapelet(X, shapelets, dist_shapelet)
