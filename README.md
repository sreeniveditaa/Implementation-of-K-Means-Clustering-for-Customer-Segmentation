# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary packages using import statement.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Import KMeans and use for loop to cluster the data.

4.Predict the cluster and plot data graphs.

5.Print the outputs and end the program

## Program:

```

Developed by: SREE NIVEDITAA SARAVANAN
RegisterNumber: 212223230213 

```

```

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib import pyplot as plt
data = pd.read_csv('/content/Mall_Customers_EX8.csv')
data
X = data[['Annual Income (k$)' , 'Spending Score (1-100)']]
X
plt.figure(figsize=(4, 4))
plt.scatter(data['Annual Income (k$)'],data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel("Spending Score (1-100)")
plt.show()
k=5
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print("Centroidsz:")
print(centroids)
print("Labels:")
print(labels)
colors = ['r','g','b','c','m']
for i in range(k):
  cluster_points = X[labels==i]
  plt.scatter(cluster_points['Annual Income (k$)'],cluster_points['Spending Score (1-100)'],color=colors[i],label=f'Cluster{i+1}')
  distances = euclidean_distances(cluster_points,[centroids[i]])
  radius = np.max(distances)
  circle = plt.Circle(centroids[i],radius,color=colors[i],fill=False)
  plt.gca().add_patch(circle)
plt.scatter(centroids[:, 0],centroids[:, 1],marker='*',s=200,color='k',label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Annual Income (k($))')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

```

## Output:

DATASET :

![Screenshot 2024-04-16 162043](https://github.com/sreeniveditaa/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/147473268/54c6fa74-1a81-44b4-a587-6ba36a90bd26)

REDUCED DATA :

![Screenshot 2024-04-16 162155](https://github.com/sreeniveditaa/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/147473268/7d299908-5a78-426b-9a08-a4437bef7922)

LABELS & CENTROIDS :

![Screenshot 2024-04-16 162314](https://github.com/sreeniveditaa/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/147473268/33a1da25-a66f-4b4b-a9b4-396f37f8df75)

K-MEANS CLUSTERING :

![Screenshot 2024-04-16 162428](https://github.com/sreeniveditaa/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/147473268/83eee6b6-5b62-4c81-91c5-2f503ae116b5)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
