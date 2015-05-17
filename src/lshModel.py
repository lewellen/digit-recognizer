from sklearn.neighbors.approximate import LSHForest
from numpy import vstack, reshape
from scipy.stats.stats import mode
from sklearn.decomposition.pca import PCA

class LSHModel:
    def __init__(self, n_neighbors = 5, n_estimators = 16):
        self.lsh = LSHForest(n_neighbors = n_neighbors, n_estimators = n_estimators)

    def fit(self, trainExamples):
        X = vstack( [reshape(x.X, (1, x.WIDTH * x.HEIGHT)) for x in trainExamples] )

        self.lsh.fit(X)
        self.Y = [x.Y for x in trainExamples]
        
        return self

    def predict(self, examples):
        X = vstack( [reshape(x.X, (1, x.WIDTH * x.HEIGHT)) for x in examples] )

        dist, ind = self.lsh.kneighbors(X)

        rows, columns = ind.shape
        for row in xrange(0, rows):
            for column in xrange(0, columns):
                ind[row, column] = self.Y[ind[row, column]]
                
        vals, counts = mode(ind, axis=1)
        
        return reshape(vals, (1, len(examples))).tolist()[0]