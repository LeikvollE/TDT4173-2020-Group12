#!/usr/bin/env python3
import pandas
from accuracy import accuracy


class mnist():
    def __init__(self):
        self.test_set = pandas.read_csv('data/mnist_test.csv')
        self.training_set = pandas.read_csv('data/mnist_test.csv')

    def get_trainig_set(self):
        return self.training_set.iloc[:, 1:].to_numpy()

    def get_test_set(self):
        return self.test_set.iloc[:, 1:].to_numpy()

    def get_accuracy(self, assignments, something='test'):
        if something == 'test':
            labels = self.test_set[['label']].to_numpy().flatten()
        else:
            labels = self.test_set[['label']].to_numpy().flatten()

        return accuracy(labels, assignments)
