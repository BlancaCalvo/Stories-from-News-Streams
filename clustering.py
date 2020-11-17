
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import pandas as pd
import os


def elbow_plot(vec_matrix_pca):
    wcss = []

    for i in range(1, 30):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=12)
        kmeans.fit(vec_matrix_pca)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 30), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')

    plt.show()

    k = int(input("Number of Clusters is: "))
    return k

def clustering_kmeans(k, vec_matrix_pca, week, reproduce = False):
    if not os.path.exists('Centroids'):
        os.mkdir('Centroids')
    if reproduce:
        centroids = np.load('Centroids/Centroids' + week + '.npy')
        clf10 = KMeans(n_clusters=k, verbose=0, random_state=12, init=centroids)
        clf10.fit(vec_matrix_pca)
    else:
        for i in range(1,3):
            if i==1:
                clf10 = KMeans(n_clusters=k, verbose=0, random_state=12, init='k-means++')
                clf10.fit(vec_matrix_pca)
                centroids = clf10.cluster_centers_
                np.save('Centroids/Centroids' + week + '.npy', clf10.cluster_centers_)
            elif i==2:
                centroids = np.load('Centroids/Centroids' + week + '.npy')
                clf10 = KMeans(n_clusters=k, verbose=0, random_state=12, init=centroids)
                clf10.fit(vec_matrix_pca)
    return clf10.cluster_centers_, clf10.labels_

def clusters_plot(centroids, labels, vec_matrix_pca, week):
    if not os.path.exists('Plots'):
        os.mkdir('Plots')
    scatter_plot_filename = 'Plots/Scatter_Plot' + week + '.png'
    plt.scatter(vec_matrix_pca[:, 0], vec_matrix_pca[:, 1], c=labels, marker='o', s=30)  # plot the documents
    plt.scatter(centroids[:, 0], centroids[:, 1], s=250, marker='+', c='red', label='Centroid')
    plt.title('Visualization of clustered data', y=1.02)
    plt.legend()
    plt.savefig(scatter_plot_filename)
    plt.show()

def cluster_names(centroids, vec_matrix_pca, data):
    closest, _ = pairwise_distances_argmin_min(centroids, vec_matrix_pca)
    central_articles = pd.DataFrame(columns=data.columns.values)
    for n, i in enumerate(closest):
        t = int(i)
        central_articles.loc[n] = data.iloc[t, :]
    central_titles = central_articles[['labels', 'TITULO']]
    central_titles.columns = ['labels', 'label_name']
    central_keywords = central_articles[['labels', 'PALABRAS_CLAVE']]
    return central_titles, central_keywords, central_articles

def silhouette_evaluation(vec_matrix_pca, labels, week):
    silhouette_vals = silhouette_samples(vec_matrix_pca, labels)  # silhouette_samples returns the value of the coefficient for every document
    y_ticks = []
    y_lower, y_upper = 0, 0
    for i, cluster in enumerate(np.unique(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]  # here we group the documents by its cluster
        cluster_silhouette_vals.sort()  # here we sort them
        y_upper += len(cluster_silhouette_vals)
        plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none',
                 height=1)  # here we make a horizontal bar plot out of it
        plt.text(-0.03, (y_lower + y_upper) / 2, str(i))  # here we call clusters by its number
        y_lower += len(cluster_silhouette_vals)

    # Get the average silhouette score and plot it
    if not os.path.exists('Plots'):
        os.mkdir('Plots')
    silhouette_plot_filename = 'Plots/Sil_Plot' + week + '.png'
    avg_score = np.mean(silhouette_vals)  # here we calculate the average score
    plt.axvline(avg_score, linestyle='--', linewidth=2, color='green')  # here we add the line of the average score
    plt.yticks([])
    plt.xlim([-0.1, 1])
    plt.xlabel('Silhouette coefficient values')
    plt.ylabel('Cluster labels')
    plt.title('Silhouette plot for the various clusters', y=1.02);
    plt.savefig(silhouette_plot_filename)
    plt.show()

def evaluation_questions(data, central_articles, central_titles):
    cl_drop = int(input("Which cluster is not coherent? (if none write 100) :"))  # if none write 100
    while cl_drop < 99:
        data = data[data['labels'] != cl_drop]
        central_articles = central_articles[central_articles['labels'] != cl_drop]
        cl_drop = int(input("Which cluster is not coherent? (if none write 100): "))  # si és que no escric 100

    changes = "yes"
    while changes == "yes":
        cl_not = int(input("Which cluster would you like to exclude? (if none write 100): "))  # si és que no escric 100
        while cl_not < 99:
            data = data[data['labels'] != cl_not]
            central_articles = central_articles[central_articles['labels'] != cl_not]
            cl_not = int(input("Which cluster would you like to exclude? (if none write 100): "))  # si és que no escric 100
        # international news are excluded
        # not clear topics are excluded
        # topic with less than 15 articles are excluded
        # non-political stories are deleted

        cl_same = int(input("Which clusters are the same? (if none write 100): "))  # si és que no escric 100
        while cl_same < 99:
            cl_same2 = int(input("With? "))
            data['labels'][data['labels'] == cl_same2] = cl_same
            cl_same = int(input("Which clusters are the same? (if none write 100): "))

        data = data.drop('label_name', axis=1)
        data = data.merge(central_titles, on='labels', how='left')
        print(data.groupby(['labels'])['DIRECCION'].size())  # canviat
        changes = input("Do you want to make any changes? Answer yes or no: ")  # yes o no
    return data