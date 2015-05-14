from sklearn.neighbors.classification import KNeighborsClassifier
from numpy import vstack, reshape

class RawModel:
    def fit(self, trainExamples):
        X = vstack( [reshape(x.X, (1, x.WIDTH * x.HEIGHT)) for x in trainExamples] )
        Y = [x.Y for x in trainExamples]
        
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.model.fit(X, Y)
        return self

    def predict(self, examples):
        X = vstack( [reshape(x.X, (1, x.WIDTH * x.HEIGHT)) for x in examples] )
        return self.model.predict( X )