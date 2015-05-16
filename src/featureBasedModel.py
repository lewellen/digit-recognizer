from copy import deepcopy
from scipy import ndimage
from sklearn import decomposition
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
from sklearn.svm.classes import SVC
from numpy import sum

class FeatureBasedModel:
    def fit(self, trainExamples):
        self.vectorizer = DictVectorizer()
        self.learner = decomposition.PCA(n_components = 15)
        self.classifier = SVC(C=1000)

        trainX = self.vectorizer.fit_transform( [ self.toDictionary(x) for x in trainExamples] )   
        trainX = self.learner.fit_transform( trainX.toarray() )
        trainY = [ int(x.Y) for x in trainExamples ]

        self.classifier.fit(trainX, trainY)
        return self

    def predict(self, examples):
        testX = self.vectorizer.transform( [ self.toDictionary(x) for x in examples ] )
        testX = self.learner.transform( testX.toarray() )
        
        return self.classifier.predict(testX)

    def toDictionary(self, example):
        # Projection
        s0 = sum(example.X, axis = 0)
        s1 = sum(example.X, axis = 1)

        a, b, c, d, e, f = sum(s0[8:10]), sum(s0[13:15]), sum(s0[18:20]), sum(s1[3:5]), sum(s1[13:15]), sum(s1[18:20])
        min0, max0, min1, max1 = min(a,b,c), max(a,b,c), min(d,e,f), max(d,e,f)

        # Symmetry
        w = example.WIDTH / 2
        h = example.HEIGHT / 2
        Q1, Q2, Q3, Q4 = example.X[:w, :h], example.X[w:, :h], example.X[:w, h:], example.X[w:, h:]
        q1, q2, q3, q4 = sum(Q1), sum(Q2), sum(Q3), sum(Q4)

        # Morphological analysis (number of closed loops)
        # http://scipy-lectures.github.io/advanced/image_processing/
        image = deepcopy(example.X)
        image = image < 32
        labeledImage, numLabels = ndimage.label(image)

        # Diagonal and antidiagonal symmetry
        rotated = ndimage.rotate(example.X, 45, reshape=False)
        R1, R2, R3, R4 = rotated[:w, :h], rotated[w:, :h], rotated[:w, h:], rotated[w:, h:]
        r1, r2, r3, r4 = sum(R1), sum(R2), sum(R3), sum(R4)

        # Rotated projection
        rot0 = sum(rotated, axis = 0)
        rot1 = sum(rotated, axis = 1)
        g, h, i, j, k, l = sum(rot0[8:10]), sum(rot0[13:15]), sum(rot0[18:20]), sum(rot1[3:5]), sum(rot1[13:15]), sum(rot1[18:20])
        rotMin0, rotMax0, rotMin1, rotMax1 = min(g, h, i), max(g, h, i), min(j, k, l), max(j, k, l)
        
        return {
                'a1': (a - min0) / float(max0 - min0),
                'a2': (b - min0) / float(max0 - min0),
                'a3': (c - min0) / float(max0 - min0),
                'b1': (d - min1) / float(max1 - min1),
                'b2': (e - min1) / float(max1 - min1),
                'b3': (f - min1) / float(max1 - min1),
                'y-symmetry': (q1 + q3) / (q2 + q4),
                'x-symmetry': (q1 + q2) / (q3 + q4),
                'closedRegions': numLabels - 1,
                'd-symmetry': (r1 + r3) / (r2 + r4),
                'ad-symmetry': (r1 + r2) / (r3 + r4),
                'c1': (g - rotMin0) / float(rotMax0 - rotMin0),
                'c2': (h - rotMin0) / float(rotMax0 - rotMin0),
                'c3': (i - rotMin0) / float(rotMax0 - rotMin0),
                'd1': (j - rotMin1) / float(rotMax1 - rotMin1),
                'd2': (k - rotMin1) / float(rotMax1 - rotMin1),
                'd3': (l - rotMin1) / float(rotMax1 - rotMin1)
                }