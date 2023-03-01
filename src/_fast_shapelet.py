

class FastShapelet:

    def __init__(self, n_shapelets, max_shapelet_length, n_jobs=1, verbose=0):
        self.n_shapelets = n_shapelets
        self.max_shapelet_length = max_shapelet_length
        self.n_jobs = n_jobs
        self.verbose = verbose
        