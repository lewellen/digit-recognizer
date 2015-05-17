from fileFormats import MNISTFormat, SubmissionFormat
from dataModel import TestExample, TrainExample
from featureBasedModel import FeatureBasedModel
from sklearn.metrics.classification import accuracy_score
from random import shuffle
from rawModel import RawModel
from lshModel import LSHModel
from datetime import datetime

def estimateAccuracy(model, limit):
    asTrain, asTest = split("../data/train.csv", limit)
     
    model.fit(asTrain)
   
    testY = [ x.Y for x in asTest ]
    testPredictions = model.predict(asTest)
  
    print("%f" % (accuracy_score(testY, testPredictions)))
    
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
    
    generateKaggleSubmission(LSHModel())
    
#    estimateAccuracy(LSHModel(), None)
    
    print("Ended: %s" % (str(datetime.now())))
