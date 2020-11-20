
import pandas as pd
import argparse
import os

from preprocess_data import preprocess, tfidf_creation
from clustering import elbow_plot, clustering_kmeans, clusters_plot, cluster_names, silhouette_evaluation, evaluation_questions

pd.options.mode.chained_assignment = None  # default='warn'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='input file path')
    parser.add_argument('--week', type=str, help='week to cluster, e.g. 2018-02 would be the third week of 2018')
    parser.add_argument('--reproduce', type=bool, default=False, help='Set to True if trying to reproduce results')
    parser.add_argument('--plots', type=bool, default=False, help='Set to True if wanting to generate the plots')
    parser.add_argument('--manual_eval', type=bool, default=True, help='Set to True to perform manual evaluation')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    data = df[(df.week == args.week)]
    data = preprocess(data)
    data.stems = [' '.join(text) for text in data.stems]

    print('This week has {0} articles.'.format(len(data)))

    vec_matrix_pca = tfidf_creation(data)

    k = elbow_plot(vec_matrix_pca)

    centroids, labels = clustering_kmeans(k, vec_matrix_pca, args.week, args.reproduce)
    data['labels'] = labels

    if args.plots:
        clusters_plot(centroids, labels, vec_matrix_pca, args.week)

    central_titles, central_keywords, central_articles = cluster_names(centroids, vec_matrix_pca, data)
    data = data.astype('object')
    data = data.merge(central_titles, on='labels', how='left')
    if not os.path.exists('Dirty'):
        os.mkdir('Dirty')
    data.to_csv('Dirty/Clusters' + args.week + '.csv', index=False)

    if args.manual_eval:
        print('Time to clean the data. Manual evaluation:')
        print(central_keywords)
        print(central_titles)
        print(data.groupby(['labels']).size())
        silhouette_evaluation(vec_matrix_pca, labels, args.week)
        data = evaluation_questions(data, central_articles, central_titles)

    new_order = [0, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    data = data[data.columns[new_order]]
    if not os.path.exists('Final'):
        os.mkdir('Final')
    data.to_csv('Final/Chosen_clusters' + args.week + '.csv', index=False)

