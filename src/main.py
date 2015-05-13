'''
Created on May 8, 2015

@author: Garrett
'''
from csv import DictReader, DictWriter
from numpy import zeros, sum, real, argmax, argmin, imag, ndarray
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn import decomposition
from numpy.dual import fft2
from scipy.ndimage.interpolation import rotate
from sklearn.svm.classes import SVC
from copy import deepcopy
from scipy import mean, ndimage
from sklearn.preprocessing.data import normalize
from scipy.stats.stats import ks_2samp


class TestExample(object):
    def __init__(self, dictionary):
        self.WIDTH = 28
        self.HEIGHT = 28

        self.X = zeros((self.WIDTH, self.HEIGHT))
        
        for key in dictionary:
            if key == "label":
                continue

            pixel = int(key.strip("pixel"))
            row = pixel / self.HEIGHT
            column = pixel % self.WIDTH
            self.X[row, column] = dictionary[key]            

    def asFeatureDict(self):
        # Projection
        s0 = sum(self.X, axis = 0)
        s1 = sum(self.X, axis = 1)
        
        a, b, c, d, e, f = sum(s0[8:10]), sum(s0[13:15]), sum(s0[18:20]), sum(s1[3:5]), sum(s1[13:15]), sum(s1[18:20])
        min0, max0, min1, max1 = min(a,b,c), max(a,b,c), min(d,e,f), max(d,e,f)

        # Symmetry
        w = self.WIDTH / 2
        h = self.HEIGHT / 2
        Q1, Q2, Q3, Q4 = self.X[:w, :h], self.X[w:, :h], self.X[:w, h:], self.X[w:, h:]
        q1, q2, q3, q4 = sum(Q1), sum(Q2), sum(Q3), sum(Q4)

        # Morphological analysis (number of closed loops)
        # http://scipy-lectures.github.io/advanced/image_processing/
        image = deepcopy(self.X)
        image = image < 32
        labeledImage, numLabels = ndimage.label(image)

        # Diagonal and antidiagonal symmetry
        rotated = ndimage.rotate(self.X, 45, reshape=False)
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

class TrainExample(TestExample):
    def __init__(self, dictionary):
        super(TrainExample, self).__init__(dictionary)
        
        self.Y = dictionary["label"]

        
class MNISTFormat:
    def deserialize(self, filePath, limit = None):
        examples = []
        with open(filePath) as handle:
            dictReader = DictReader(handle, delimiter=',')
            for example in dictReader:
                examples.append(example)
                if limit != None:
                    if limit <= 0:
                        break
                    limit -= 1

        return examples
    
class SubmissionFormat:
    def serialize(self, filePath, Y):
        with open(filePath, 'wb') as handle:
            handle.write("ImageId,Label\n")
            handle.writelines(["%d,%s\n" % (i,y) for (i, y) in enumerate(Y, start=1)])

if __name__ == '__main__':
    limit = 10000
    fileFormat = MNISTFormat()
    train =  [TrainExample(x) for x in fileFormat.deserialize("../data/train.csv", limit)]
    limit = len(train)

    M = int(0.8 * limit)
    
    pca = decomposition.PCA(n_components = 15)

    vectorizer = DictVectorizer()
    asTraining = train[:M]
    trainX = vectorizer.fit_transform( [ x.asFeatureDict() for x in  asTraining] )   
    trainX = pca.fit_transform( trainX.toarray() )
    trainY = [ int(x.Y) for x in asTraining ]

#    classifier = KNeighborsClassifier(n_neighbors = 7)
    classifier = SVC(C=1000)
    classifier.fit(trainX, trainY)

    asTest = train[M:]
    testX = vectorizer.transform( [ x.asFeatureDict() for x in asTest ] )
    testX = pca.transform( testX.toarray() )
    testY = [ int(x.Y) for x in asTest ]

    testPredictions = classifier.predict(testX)

    for (x, y, p) in zip(testX, testY, testPredictions):
        print("%f %f %f %f %f" % (x[0], x[1], x[2], y, p))

    guess = [TestExample(x) for x in fileFormat.deserialize("../data/test.csv")]
    guessX = vectorizer.transform( [ x.asFeatureDict() for x in guess ]  )
    guessX = pca.transform( guessX.toarray() )
    guessPredictions = classifier.predict(guessX)
   
    outputFormat = SubmissionFormat()
    outputFormat.serialize("../data/guesses.csv", guessPredictions)