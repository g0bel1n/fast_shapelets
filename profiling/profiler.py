import sys 
import os

if 'fast_shapelets' not in [el.split('/')[-1] for el in sys.path]:
    curr_path = os.getcwd()
    sys.path.append('/'.join((curr_path.split('/')[:-1])))


from src import get_dataset, SAX, FastShapelet
import numpy as np
import matplotlib.pyplot as plt


X_train,y_train, X_test, y_test = get_dataset('StarLightCurves')
y_train = y_train-1
y_test = y_test-1
fs = FastShapelet(11, 203, 202, n_jobs=1, verbose=1)
fs.fit(X_train[:100], y_train[:100])