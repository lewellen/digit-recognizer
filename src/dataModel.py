from numpy import zeros
import numpy

class TestExample(object):
    def __init__(self, dictionary):
        self.WIDTH = 28
        self.HEIGHT = 28

        # 2015-05-15 GEL Changed dtype=numpy.uint8 (0, 255) to reduce memory.
        self.X = zeros((self.WIDTH, self.HEIGHT), dtype=numpy.uint8)
        
        for key in dictionary:
            if key == "label":
                continue

            pixel = int(key.strip("pixel"))
            row = pixel / self.HEIGHT
            column = pixel % self.WIDTH
            self.X[row, column] = dictionary[key]            

class TrainExample(TestExample):
    def __init__(self, dictionary):
        super(TrainExample, self).__init__(dictionary)
        
        self.Y = int(dictionary["label"])