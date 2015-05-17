from sklearn.neighbors.classification import KNeighborsClassifier
from numpy import vstack, reshape, logical_or
from sklearn.decomposition.truncated_svd import TruncatedSVD
from sklearn.ensemble.bagging import BaseBagging

class PatchedRawModel:
    def __init__(self):
        self.baseModel = RawModel()
        self.model49 = KNeighborsClassifier(n_neighbors=10)
        self.model35 = KNeighborsClassifier(n_neighbors=10)
    
    def fit(self, trainExamples):
        self.baseModel.fit(trainExamples)

        X49 = vstack ( [reshape(x.X, (1, x.WIDTH * x.HEIGHT)) for x in trainExamples if x.Y in [4, 9]] )
        Y49 = [x.Y for x in trainExamples if x.Y in [4, 9]]
        self.model49.fit(X49, Y49)

        X35 = vstack ( [reshape(x.X, (1, x.WIDTH * x.HEIGHT)) for x in trainExamples if x.Y in [3, 5]] )
        Y35 = [x.Y for x in trainExamples if x.Y in [3, 5]]
        self.model35.fit(X35, Y35)

    def predict(self, examples):
        basePredictions = self.baseModel.predict(examples)

        for (x, y, i) in zip(examples, basePredictions, range(0, len(examples))):
            if y in [4, 9]:
                specializedPrediction = self.model49.predict(reshape(x.X, (1, x.WIDTH * x.HEIGHT)))
                if specializedPrediction != y:
                    basePredictions[i] = specializedPrediction
            elif y in [3, 5]:
                specializedPrediction = self.model35.predict(reshape(x.X, (1, x.WIDTH * x.HEIGHT)))
                if specializedPrediction != y:
                    basePredictions[i] = specializedPrediction

        return basePredictions

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