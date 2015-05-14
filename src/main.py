from fileFormats import MNISTFormat, SubmissionFormat
from dataModel import TestExample, TrainExample
from featureBasedModel import FeatureBasedModel
from sklearn.metrics.classification import accuracy_score
from random import shuffle
from sklearn.neighbors.classification import KNeighborsClassifier
from numpy import reshape, vstack
from sklearn.manifold import isomap
from sklearn import lda
from sklearn.decomposition import pca
from rawModel import RawModel

def split(filePath, limit):
    fileFormat = MNISTFormat()
    examples =  [ TrainExample(x) for x in fileFormat.deserialize(filePath, limit) ]
    shuffle(examples)
    M = int(0.8 * len(examples))

    return examples[:M], examples[M:]

if __name__ == '__main__':
    asTrain, asTest = split("../data/train.csv", limit=10000)

#     model = FeatureBasedModel()
    model = RawModel()
    model.fit(asTrain)
 
    testY = [ x.Y for x in asTest ]
    testPredictions = model.predict(asTest)

    print("Accuracy: %f" % accuracy_score(testY, testPredictions))

    fileFormat = MNISTFormat()
    guess =  [ TestExample(x) for x in fileFormat.deserialize("../data/test.csv") ]
    guessPredictions = model.predict(guess)
     
    outputFormat = SubmissionFormat()
    outputFormat.serialize("../data/guesses.csv", guessPredictions)