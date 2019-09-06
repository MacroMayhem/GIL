__author__ = "Aditya Singh"
__version__ = "0.1"


class Metrics:
    def __init__(self, **kwargs):
        self.metrics = kwargs.get('metrics', ['acc', 'bwt', 'fwt'])

    def accuracy(self):
        pass

    def backward_transfer(self):
        pass

    def forward_transfer(self):
        pass
