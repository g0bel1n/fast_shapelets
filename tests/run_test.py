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
    from fast_shapelets.src._utils import DTW_distance

    X_train,y_train, X_test, y_test = get_dataset('StarLightCurves')
    y_train = y_train-1
    y_test = y_test-1

    X = X_train[:50]
    y = y_train[:50]

    # %%
    fs = FastShapelets(min_shapelet_length=200, max_shapelet_length=200, cardinality=4, dimensionality=16, r=5)

    # %%
    fs.fit(X, y, dist_shapelet = DTW_distance)

    fs.transform(X_train[:10])
# %%
