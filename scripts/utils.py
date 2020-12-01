from nltk.tokenize import word_tokenize
import gensim
import pandas as pd

def extract_tfidf_per_cluster(data):
    list_of_df = []
    for label in data['labels'].unique():
        df = data[data['labels'] == label]
        list_of_df.append(df)

    list_of_stems_df = []
    list_of_labels = []

    for df in list_of_df:
        string_with_text = ""
        for index, row in df.iterrows():
            stems = str(row['stems'])
            string_with_text += stems
        label = df['labels'].unique()
        list_of_labels.append(label)
        list_of_stems_df.append(string_with_text)

    gen_docs = [[w for w in word_tokenize(text)]
                for text in list_of_stems_df]

    dictionary = gensim.corpora.Dictionary(gen_docs)

    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
    tf_idf = gensim.models.TfidfModel(corpus)
    return tf_idf, corpus, dictionary, list_of_labels

def compare_to_week(data, tf_idf, corpus, dictionary, list_of_labels, threshold, min_articles):
    sims = gensim.similarities.Similarity('.', tf_idf[corpus], num_features=len(dictionary))
    results = pd.DataFrame()
    for n, row in data.iterrows():
        article = str(row['stems'])  # this line is to avoid the nan values
        query_doc = [w for w in word_tokenize(article)]
        query_doc_bow = dictionary.doc2bow(query_doc)  # doc2vow converts our corpus to bag-of-words
        query_doc_tf_idf = tf_idf[query_doc_bow]
        sims_list = list(sims[query_doc_tf_idf])
        result = pd.DataFrame(
            {'labels_new': list_of_labels, 'similarity': sims_list, 'DIRECCION': data.iloc[n]['DIRECCION']})
        results = pd.concat([results, result])

    results.labels_new = results.labels_new.map(int)
    results = results[results['similarity'] > threshold]
    grouped = results.groupby(['labels_new'])
    results = grouped.filter(lambda x: x['DIRECCION'].count() > min_articles)

    new = results.merge(data, how='inner', on='DIRECCION')
    new = new.drop('labels', axis=1)
    new.rename(columns={'labels_new': 'labels'}, inplace=True)

    return new

def merge_weeks(origin, new, finalized_df):
    new = pd.concat([origin, new], ignore_index=True, sort=False)
    names = new['labels'].unique()
    #if first_time:
    #    numbers = range(0, len(names))
    #    first_time = False
    #else:
    numbers = range(int(max(finalized_df['labels'])) + 1, int(max(finalized_df['labels'])) + len(names) + 1)
    standing_labels = pd.DataFrame({'labels': names, 'final_labels': numbers})
    forward = new.merge(standing_labels, on='labels')

    forward = forward.drop('labels', axis=1)
    forward.rename(columns={'final_labels': 'labels'}, inplace=True)

    return forward

def compare_and_merge_weeks(origin, compare_to, similarity_threshold, min_articles, finalized_df):
    tf_idf, corpus, dictionary, list_of_labels = extract_tfidf_per_cluster(origin)
    new = compare_to_week(compare_to, tf_idf, corpus, dictionary, list_of_labels, similarity_threshold, min_articles)
    if new.shape[0] > 0:
        forward = merge_weeks(origin, new, finalized_df)
    else:
        forward = origin
    return forward, new

def check_clusters_and_save(finalized_df, forward, merging_thrshold):
    for i in finalized_df['labels'].unique():
        previous = finalized_df[finalized_df['labels'] == i]
        for d in forward['labels'].unique():
            now = forward[forward['labels'] == d]
            s1 = pd.merge(previous, now, how='inner', on=['DIRECCION'])
            value = len(s1) / len(now)  # calculates the percentage of articles in the new clusters that are already in a previous cluster
            value2 = len(s1) / len(previous)  # calucaltes the percentage of articles in the new clusters that are already in a previous cluster
            if value > merging_thrshold:  # if it's bigger than 0.4 then the articles in the new cluster get the label of the previous cluster, they become the same
                print('Label {0} becomes label {1} for a similarity of {2}.'.format(d,i,value))
                forward['labels'] = forward['labels'].replace(d, i)
            elif value2 > merging_thrshold:
                print('Label {0} becomes label {1} for a similarity of {2}.'.format(i,d,value2))
                finalized_df['labels'] = finalized_df['labels'].replace(i, d)

    finalized_df = pd.concat([finalized_df, forward])
    finalized_df.to_csv('corpus.csv', index=False)
    return finalized_df