import numpy as np

def load_stars():
    stars = np.genfromtxt('data/stars.csv', delimiter=',', skip_header=1)
    return stars[:,:-3]
