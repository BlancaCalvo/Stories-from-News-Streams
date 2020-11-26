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
    print(origin.head())
    print(new.head())
    new = pd.concat([origin, new], ignore_index=True, sort=False)
    print(new.head())
    names = new['labels'].unique()
    #if first_time:
    #    numbers = range(0, len(names))
    #    first_time = False
    #else:
    print(finalized_df)
    numbers = range(int(max(finalized_df['final_labels'])) + 1, int(max(finalized_df['final_labels'])) + len(names) + 1)
    standing_labels = pd.DataFrame({'labels': names, 'final_labels': numbers})
    print(standing_labels)

    forward = new.merge(standing_labels, on='labels')

    #forward.drop('labels', axis=1)
    #forward.rename(columns={'final_labels': 'labels'}, inplace=True)

    print(forward.head())
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
    for i in finalized_df['final_labels'].unique():
        previous = finalized_df[finalized_df['final_labels'] == i]
        for d in forward['final_labels'].unique():
            now = forward[forward['final_labels'] == d]
            s1 = pd.merge(previous, now, how='inner', on=['DIRECCION'])
            value = len(s1) / len(
                now)  # calculates the percentage of articles in the new clusters that are already in a previous cluster
            print(value)
            value2 = len(s1) / len(
                previous)  # calucaltes the percentage of articles in the new clusters that are already in a previous cluster
            print(value2)
            if value > merging_thrshold:  # if it's bigger than 0.4 then the articles in the new cluster get the label of the previous cluster, they become the same
                print('1.Label ' + str(d) + ' becomes label ' + str(i))
                forward['final_labels'] = forward['final_labels'].replace(d, i)
            elif value2 > merging_thrshold:
                print('2.Label ' + str(i) + ' becomes label ' + str(d))
                finalized_df['final_labels'] = finalized_df['final_labels'].replace(i, d)

    finalized_df = pd.concat([finalized_df, forward])
    return finalized_df