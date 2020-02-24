import numpy as np


class Statistics:

    def __init__(self, keep_all=False):
        self.keep_all = keep_all
        self.data = []
        self.n = 0
        self.avg = 0.0

    def add(self, x):
        if self.keep_all:
            self.data += x
        else:
            self.avg = self.avg * (self.n / float(self.n + 1)) + x * (1. / float(self.n + 1))
            self.n += 1

    def mean(self):
        if self.keep_all:
            return np.mean(self.data)
        else:
            return self.avg

    @staticmethod
    def get_stats(k, keep_all=False):
        return [Statistics(keep_all) for _ in range(k)]

    def np(self):
        return np.array(self.data)
