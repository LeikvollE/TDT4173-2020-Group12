#!/usr/bin/env python3
import pandas


class weka():
    def __init__(self):
        self.two = pandas.read_csv('data/column_2C_weka.csv')
        self.three = pandas.read_csv('data/column_3C_weka.csv')

    def get_two(self):
        return self.two.iloc[:, 0:-1].to_numpy()

    def get_three(self):
        return self.three.iloc[:, 0:-1].to_numpy()
