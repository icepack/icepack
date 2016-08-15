
class GridData(object):
    def __init__(self, _x, _y, _data, _missing):
        self.x = _x
        self.y = _y
        self.data = _data
        self.missing = _missing

        #TODO: add an exception if the sizes of x, y, data don't match up
        #TODO: use numpy masked array
