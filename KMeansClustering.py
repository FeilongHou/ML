# 1.Randomly place k centroids in random postions
# in the graph. 
# 2.Draw straight line between 2 centroidd
# 3. then draw a line passing through the midpoint of that
# straight line. 
# 4. One side belong to one centroid.
# In other word, assign points to cloest centroid

# We find the middle of points belong to one centroid
# Then put that centroid there
# The way we find the middle points is: sum(x 1 to n)/n
# repeat until no changing in data points
# p*C*i*features calculations

# This is a unsupervised learning
import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()
# scale all the data down to {-1,1}
data = scale(digits.data)
# the targets are a array of number that, for example, [0,1,2,3,4,2,6,9,7,8,5]
y = digits.target

# set K to be number of unique targets in y = 10
k = len(np.unique(y))
samples, features = data.shape


def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

# init define how to place centroids. "random" is random, "k-means++" give more sensiable placement
# n_init is times that algorithm will run with different centroid
# max_iter is max interation default = 300
clf = KMeans(n_clusters=k, init = "random", n_init = 10)
bench_k_means(clf, "1", data)