from numpy import zeros

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

class TrainExample(TestExample):
    def __init__(self, dictionary):
        super(TrainExample, self).__init__(dictionary)
        
        self.Y = dictionary["label"]