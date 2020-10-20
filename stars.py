import numpy as np

def load_stars():
    stars = np.genfromtxt('data/stars.csv', delimiter=',', skip_header=1)
    stars = stars[:,:-3]
    return (stars - np.min(stars, axis=0)) / (np.max(stars) - np.min(stars))
