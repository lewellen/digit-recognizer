from fileFormats import MNISTFormat, SubmissionFormat
from dataModel import TestExample, TrainExample
from featureBasedModel import FeatureBasedModel
from sklearn.metrics.classification import accuracy_score, confusion_matrix
from random import shuffle
from rawModel import RawModel, PatchedRawModel
from lshModel import LSHModel
from datetime import datetime
from numpy import vstack, reshape
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.neighbors.classification import KNeighborsClassifier
from nltk.classify.naivebayes import NaiveBayesClassifier
from sklearn.svm.classes import SVC
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.manifold.isomap import Isomap
from sklearn.decomposition.pca import PCA

def estimateAccuracy(model, limit):
    asTrain, asTest = split("../data/train.csv", limit)
     
    model.fit(asTrain)
   
    testY = [ x.Y for x in asTest ]
    testPredictions = model.predict(asTest)
  
    print("%f" % (accuracy_score(testY, testPredictions)))
    
    print confusion_matrix(testY, testPredictions)
    
def generateKaggleSubmission(model):
    fileFormat = MNISTFormat()
    examples =  [ TrainExample(x) for x in fileFormat.deserialize("../data/train.csv", limit=None) ]

    model.fit(examples)

    guess =  [ TestExample(x) for x in fileFormat.deserialize("../data/test.csv") ]
    guessPredictions = model.predict(guess)
       
    outputFormat = SubmissionFormat()
    outputFormat.serialize("../data/guesses.csv", guessPredictions)

def split(filePath, limit):
    fileFormat = MNISTFormat()
    examples =  [ TrainExample(x) for x in fileFormat.deserialize(filePath, limit) ]
    shuffle(examples)

    M = int(0.8 * len(examples))

    return examples[:M], examples[M:]

if __name__ == '__main__':  
    print("Began: %s" % (str(datetime.now())))

    generateKaggleSubmission(PatchedRawModel())
     
#     estimateAccuracy(PatchedRawModel(), None)
     
    print("Ended: %s" % (str(datetime.now())))
