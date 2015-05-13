from csv import DictReader

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