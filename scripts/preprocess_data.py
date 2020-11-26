
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def remove_stopwords(words):
    important_words = []
    for word in words:
        if word not in stopwords.words('spanish'):
            important_words.append(word)
    return important_words

def preprocess(data):
    print('Preprocessing data...')
    data.dropna(subset=['TEXTO'], inplace=True)
    stemmer = SnowballStemmer('spanish')
    data['stems'] = data.TEXTO.apply(lambda x: [stemmer.stem(i) for i in word_tokenize(x)])
    #data.stems = data.stems.apply(lambda x: [remove_stopwords(i) for i in x])
    return data

def tfidf_creation(data):
    print('Creating TF-IDF matrix...')
    vec = TfidfVectorizer(stop_words=stopwords.words('spanish'), tokenizer=word_tokenize)
    vec_matrix = vec.fit_transform(data.stems)
    pca = TruncatedSVD(n_components=100)
    vec_matrix_pca = pca.fit_transform(vec_matrix)
    return vec_matrix_pca


