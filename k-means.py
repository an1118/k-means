# -*- coding: utf-8 -*-

# DSCI 552: Machine Learning for Data Science
# Programming Assignment 2 PART 1: k-means
# Team members: Li An, Shengnan Ke


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def kmeansSingleRun(data, k):
  # create a np array to save cluster for each data point
  clusters = np.zeros(data.shape[0])
  # initialize k centroids
  centroids_ary = data.sample(n=k).to_numpy()
  # while the algorithm doesn't converge -> centroids still changing
  flag = 1
  while flag:
    # assign each data point to its closest centroid
    # for each data point
    for index, row in data.iterrows():
      x1 = row[0]
      x2 = row[1]
      min_dist = float('inf')  # initialize minimum distance as an infinite number
      cluster = None
      # compute distance of the data point from all centroids, find the closest one
      for idx, centroid in enumerate(centroids_ary):
        dist = np.sqrt((x1 - centroid[0]) ** 2 + (x2 - centroid[1]) ** 2)
        if dist < min_dist:
          min_dist = dist
          cluster = idx
      clusters[index] = cluster
    # recompute centroids for each group
    new_centroids_ary = data.groupby(by=clusters).mean().values
    # compare to see if centroids stopped changing
    if (new_centroids_ary==centroids_ary).all():
      flag = 0
    else:
      centroids_ary = new_centroids_ary
  return centroids_ary, clusters


def clustersPlot(data, clusters, centroids):
  sns.scatterplot(data.loc[:, 0], data.loc[:, 1], hue=clusters,  palette='tab10')
  sns.scatterplot(centroids[:, 0], centroids[:, 1], s=100, color='red')
  plt.title("K-means clustering result")
  plt.xlabel('x1')
  plt.ylabel('x2')
  plt.show()
  return


# load data
df_data = pd.read_csv('clusters.txt', sep=',', header=None)

# run k-means
k = 3
centroids, clusters = kmeansSingleRun(df_data, k)
clusters = clusters.astype(int)
# clusters, clusters_n = kmeansMultiTimes(df_data, k, 5)
clustersPlot(df_data, clusters, centroids)

