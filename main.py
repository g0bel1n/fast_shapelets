if __name__ == '__main__' :
    import sys 
    import os
    import pickle
    import argparse
        
    
    if 'fast_shapelets' not in [el.split('/')[-1] for el in sys.path]:
        curr_path = os.getcwd()
        sys.path.append('/'.join((curr_path.split('/')[:-1])))
        
    from src import get_dataset, FastShapelets
    import numpy as np
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser()
    # Dataset and dataloader
    parser.add_argument("--n_jobs", type=int, default=1, help="n jobs")
    parser.add_argument("--min_shap_len", type=int, default=100, help="min_shap_len")
    parser.add_argument("--max_shap_len", type=int, default=500, help="max_shap_len")
    parser.add_argument("--step_shap_len", type=int, default=100, help="step_shap_len")
    parser.add_argument("--cardinality", type=int, default=4, help="cardinality")
    parser.add_argument("--dimensionality", type=int, default=16, help="dimensionality")
    parser.add_argument("--r", type=int, default=10, help="r")
    parser.add_argument("--verbose", type=int, default=2, help="verbose")


    
    
    args = parser.parse_args()
    
    X_train,y_train, X_test, y_test = get_dataset('StarLightCurves')
    y_train = y_train-1
    y_test = y_test-1
        
    shapelet_lengths = list(range(args.min_shap_len,args.max_shap_len+1, args.step_shap_len))
    fs = FastShapelets(shapelet_lengths=shapelet_lengths, cardinality=args.cardinality, dimensionality=args.dimensionality, r=args.r, n_jobs=args.n_jobs, verbose=args.verbose)
    fs.fit(X_train, y_train)
    
    fn_save = f'shap_{args.min_shap_len}_{args.max_shap_len+1}_{args.step_shap_len}.pkl'
    with open(fn_save, 'wb') as f:
        pickle.dump(fs.get_shapelets(), f)
        
    
    print(f'file saved at {fn_save}')
        
    