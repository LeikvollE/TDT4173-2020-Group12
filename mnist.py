#!/usr/bin/env python3
import pandas
import numpy as np


class mnist():
    def __init__(self):
        self.test_set = pandas.read_csv('data/mnist_test.csv')
        self.test_set = (self.test_set - np.min(self.test_set)) / (np.max(self.test_set) - np.min(self.test_set) + 0.0001)
        self.training_set = pandas.read_csv('data/mnist_test.csv')

    def get_trainig_set(self):
        return self.training_set.iloc[:, 1:].to_numpy()

    def get_test_set(self):
        return self.test_set.iloc[:, 1:].to_numpy()
