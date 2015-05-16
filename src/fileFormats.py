from csv import DictReader

class MNISTFormat:
    def deserialize(self, filePath, limit = None):
        # 2015-05-15 GEL Changed to a generator so that the whole file isn't 
        # loaded into memory as other data structures are being constructed.
        with open(filePath) as handle:
            dictReader = DictReader(handle, delimiter=',')
            for example in dictReader:
                yield example
                if limit != None:
                    if limit <= 0:
                        break
                    limit -= 1
    
class SubmissionFormat:
    def serialize(self, filePath, Y):
        with open(filePath, 'wb') as handle:
            handle.write("ImageId,Label\n")
            handle.writelines(["%d,%s\n" % (i,y) for (i, y) in enumerate(Y, start=1)])