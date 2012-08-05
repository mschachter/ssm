
class InputStream(object):

    def __init__(self):
        self.id = None
        self.length = 0

    def __len__(self):
        return self.length

    def next(self, t):
        pass

class InputCurrentStream(InputStream):

    def __init__(self, ifunc):
        InputStream.__init__(self)
        self.length = 1
        self.ifunc = ifunc

    def next(self, t):
        return self.ifunc(t)



