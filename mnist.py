#!/usr/bin/env python3
import pandas
<<<<<<< HEAD
import numpy as np
=======
from accuracy import accuracy
>>>>>>> bf283af7697d4214f978787578b2d5b5ecb17b33


class mnist():
    def __init__(self):
        self.test_set = pandas.read_csv('data/mnist_test.csv')
        self.test_set = (self.test_set - np.min(self.test_set)) / (np.max(self.test_set) - np.min(self.test_set) + 0.0001)
        self.training_set = pandas.read_csv('data/mnist_test.csv')

    def get_trainig_set(self):
        return self.training_set.iloc[:, 1:].to_numpy()

    def get_test_set(self):
        return self.test_set.iloc[:, 1:].to_numpy()

    def get_accuracy(self, assignments, something='test'):
        if something == 'test':
            labels = self.test_set[['label']].to_numpy().flatten()
        else:
            labels = self.training_set[['label']].to_numpy().flatten()

        return accuracy(labels, assignments)
