import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits =load_digits()
#data is all our featurs and we will sacle them down so they are between -1 to 1 we are doing this because by default the vaule's digits are large
#scaling them down will also save time when calculating
data  = scale(digits.data)
#labels

y = digits.target

#clusters

#this if for scaling
#you could also just type k  = 10 for 10 digits
k = len(np.unique(y))

samples , features = data.shape

#scoring system

#sklearn function is from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
#we are just taking the scoring system from sklearn's website
#in the end we will be able to train a lot of diffrent classifyers since we can just use the function multiple times
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
#the amount of centroid's ( its = to K
#where it will place the centros ( in this case its random )
#we can run it 10 times and it will take the best classifyers ( 10 is default )
#the maxium initialization default is 300
#can read more on : https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html?highlight=sklearn%20cluster%20kmeans#sklearn.cluster.KMeans
clf = KMeans(n_clusters=k, init="random", n_init=10)

bench_k_means(clf,"1",data)