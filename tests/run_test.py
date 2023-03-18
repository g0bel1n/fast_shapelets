# %%




if __name__ == "__main__":
# %%
    import sys 
    import os

    if 'fast_shapelets' not in [el.split('/')[-1] for el in sys.path]:
        curr_path = os.getcwd()

        sys.path.append('/'.join((curr_path.split('/')[:-1])))


    from fast_shapelets.src import get_dataset, FastShapelets
    import numpy as np
    from fast_shapelets.src._utils import dtw_jax, _DTW,DTW_distance

    X_train,y_train, X_test, y_test = get_dataset('StarLightCurves')
    y_train = y_train-1
    y_test = y_test-1

    X = X_train[:500]
    y = y_train[:500]

    # %%
    fs = FastShapelets(min_shapelet_length=200, max_shapelet_length=200, cardinality=4, dimensionality=16, r=10)

    # %%
    fs.fit(X, y, dist_shapelet = DTW_distance)

    # %%
    test_shap = fs.transform(X_test[:500])
    train_shap = fs.transform(X_train[:500])

    # %%
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import balanced_accuracy_score


    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_shap, y_train[:500])
    knn.score(test_shap, y_test[:500])
#balanced_accuracy_score(y_test[:1000], knn.predict(test_shap))

#%%
import sys 
import os

if 'fast_shapelets' not in [el.split('/')[-1] for el in sys.path]:
    curr_path = os.getcwd()

    sys.path.append('/'.join((curr_path.split('/')[:-1])))
# %%
from src._utils import dtw_jax

dtw_jax(np.array([1,2,3,4,5,6,7,8,9,10]), np.array([1,2,3,4,5,6,7,8,9,10]))
# %%
dtw(np.array([1,2,3,4,5,6,7,8,9,10]), np.array([1,2,3,4,5,6,7,8,9,10]))
# %%
