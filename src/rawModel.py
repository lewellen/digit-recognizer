from sklearn.neighbors.classification import KNeighborsClassifier
from numpy import vstack, reshape
from sklearn.decomposition.truncated_svd import TruncatedSVD

class RawModel:
    def __init__(self):
        # 2015-05-15 GEL Found that n_components=20 gives a nice balance of 
        # speed (substantial improvement), accuracy, and reduced memory usage 
        # (25% decrease).
        self.decomposer = TruncatedSVD(n_components=20)

        # 2015-05-15 GEL algorithm='ball_tree' uses less memory on average than 
        # algorithm='kd_tree'
        
        # 2015-05-15 GEL Evaluation of metrics by accuracy (based on 8000 training examples)
        # euclidean        0.950025
        # manhattan        0.933533
        # chebyshev        0.675662
        # hamming          0.708646
        # canberra         0.934033
        # braycurtis       0.940530
        self.model = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree', metric='euclidean')

    def fit(self, trainExamples):
        X = self.decomposer.fit_transform( vstack( [reshape(x.X, (1, x.WIDTH * x.HEIGHT)) for x in trainExamples] ) )
        Y = [x.Y for x in trainExamples]

        self.model.fit(X, Y)
        return self

    def predict(self, examples):
        X = self.decomposer.transform( vstack( [reshape(x.X, (1, x.WIDTH * x.HEIGHT)) for x in examples] ) )
        return self.model.predict( X )