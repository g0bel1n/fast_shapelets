# %%



if __name__ == '__main__':
    import sys 
    import os

    if 'fast_shapelets' not in [el.split('/')[-1] for el in sys.path]:
        curr_path = os.getcwd()

        sys.path.append('/'.join((curr_path.split('/')[:-1])))


    from src import get_dataset, FastShapelets
    import numpy as np
    from src._utils import DTW_distance
    from multiprocessing import cpu_count
    import time

    X_train,y_train, X_test, y_test = get_dataset('StarLightCurves')
    y_train = y_train-1
    y_test = y_test-1

    X = X_train[:200]
    y = y_train[:200]

    n_jobs = 5
    fs = FastShapelets(min_shapelet_length=200, max_shapelet_length=200, cardinality=4, dimensionality=16, r=10, n_jobs=n_jobs)

    #%%
    t = time.perf_counter()
    times = [t]
    for _ in range(1):
        fs.fit(X, y)
        times.append(time.perf_counter())


    times = np.array(times)
    times = times[1:]-times[:-1]
    print(f'Average time: {np.mean(times)}')
    print(f'Max time: {np.max(times)}')
    print(f'Min time: {np.min(times)}')
    print(f'Std time: {np.std(times)}')

    #%%
    fs.transform(X_train[:10])
# %%
