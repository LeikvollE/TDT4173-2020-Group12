# TDT4173-2020-Group12

This repository contains code and data used in our analysis of k-means' performance when augmented with popular modifications:

# Try it yourself!
The jupyter notebook Demo.ipynb demonstrates use of the code we made for our project. Please read through this README before trying yourself.

# Files and folders:

The data folder contains the star dataset in stars.csv. MNIST is not included due to its size, but can be downloaded in .csv format and placed in the data folder from this link: https://www.kaggle.com/oddrationale/mnist-in-csv

In the distance folder the file distance_functions.py includes both a manhattan and Euclidean distance function

The kmeans folder includes two implementations of kmeans. In basic.py, the random vector and Forgy (aka random sample) initialisation methods can be used by way of the following lines in the kmeans() function:

```
centroids = np.random.normal(0.0, 1.0, [k, len(data[0])]) # random vector
# centroids = data[random.sample(range(len(data)), k)].copy() # forgy
```

The kmeanspp.py file includes kmeans++ with its variation on cluster initialisation. For both kmeans and kmeans++, the distance function can be changed in one line: either use

```
dist_j = -1
   for j, centroid in enumerate(centroids):
       new_dist = euclidean_dist(entry, centroid)
```
or
```
dist_j = -1
   for j, centroid in enumerate(centroids):
       new_dist = manhattan_dist(entry, centroid)
```

The jupyter notebook Demo.ipynb demonstrates use of our code. By switching the commenting in section \[6\] of the notebook, PCA can be used on MNIST, courtesy of scikit-learn.

The files mnist.py and stars.py load and prepare thedatses. The main function in stars.py computes the heuristics which were highlighted in our podcast.

The best of n runs enhancement is set in the last parameter of the kmeans and kmeanspp functions (in the notebook n is currently set to 1).
