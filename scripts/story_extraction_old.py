import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import gensim

a = "2018-"

n_weeks = 51 # last week linked

threshold = 0.35 # similarity threshold

min_articles = 10 # minimum of articles per week to move on

fusion_threshold = 0.33 # percentage of articles that have to be the same
# he canviat això, però també podria posar un threshold de fusió proporcional al número de articles al cluster, però no acabo de veure com


b = 0 
counter = "%02d" % (int(b))
starting = a + str(counter)
print(starting)

c = b + 1
counter = "%02d" % (int(c))
following = a + str(counter)
print(following)

data1 = pd.read_csv("Final/Chosen_clusters"+starting+".csv")
data2 = pd.read_csv("Results/Divisio"+following+".csv")

pd.options.display.max_colwidth = 100 # configurar quants caràcters imprimir al fer print() a una columna de strings

# from every topic in the starting csv create a string of stems
print(len(data1))
list_of_df = []

for label in data1['labels'].unique():
	df = data1[data1['labels']==label]
	list_of_df.append(df)

# for every df take the column stems and do tokens out of them, then calculate tf-idf, keep a df that just contains stems and tf-idf values
list_of_stems_df = []
list_of_labels = []

for df in list_of_df:
	string_with_text = ""
	for row in df:
		stems = str(df['stems'])
		string_with_text+=stems
	label = df['labels'].unique()
	list_of_labels.append(label)
	list_of_stems_df.append(string_with_text)

gen_docs = [[w for w in word_tokenize(text)] 
            for text in list_of_stems_df]

dictionary = gensim.corpora.Dictionary(gen_docs)

corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
tf_idf = gensim.models.TfidfModel(corpus)

sims = gensim.similarities.Similarity('.',tf_idf[corpus], num_features=len(dictionary)) # here we create the model to evaluate similarity

# we run our model to compute similarities with all the articles from the following week
print(len(data2))
results = pd.DataFrame()
for n,article in enumerate(data2['stems']):
	article = str(article) # this line is to avoid the nan values
	query_doc = [w for w in word_tokenize(article)]
	query_doc_bow = dictionary.doc2bow(query_doc) # doc2vow converts our corpus to bag-of-words
	query_doc_tf_idf = tf_idf[query_doc_bow]
	sims_list = list(sims[query_doc_tf_idf])
	result = pd.DataFrame({'labels_new':list_of_labels, 'similarity': sims_list, 'DIRECCION':data2.iloc[n]['DIRECCION']})
	results = pd.concat([results,result])

results.labels_new = results.labels_new.map(int)
results = results[results['similarity']>threshold]
grouped = results.groupby(['labels_new'])
results = grouped.filter(lambda x: x['DIRECCION'].count() > min_articles) # checks that more than 7 news in the new month have been detected

new = results.merge(data2, how = 'inner', on = 'DIRECCION')
new = new.drop('labels', axis = 1)
new.rename(columns={'labels_new':'labels'}, inplace=True)
print(len(new))

new2 = pd.concat([data1, new], ignore_index=True, sort = False)
print(len(new2))

names = new2['labels'].unique()
numbers = range(0,len(names))
df1 = pd.DataFrame({'labels':names, 'final_labels': numbers})
print(df1)

final = new2.merge(df1, on = 'labels')

# CHECK NOW FOR WEEK 2 UNTIL THE END
if len(results) >= min_articles:
	while c < n_weeks: # when we have the data this should be 52
		c = c + 1
	#print(c)
		counter = "%02d" % (int(c))
		following = a + str(counter)
		print(following)
		data2 = pd.read_csv("Results/Divisio"+following+".csv")
	# filter data1 by the labels that are included in results
		continuing_labels = results['labels_new'].unique()
		data1 = data1.loc[data1['labels'].isin(continuing_labels)] 
		print(data1['labels'].unique())
	# CREATE A NEW MODEL WITH THE RAMAINING LABELS
		list_of_df = []
		for label in data1['labels'].unique():
			df = data1[data1['labels']==label]
			list_of_df.append(df)
		list_of_stems_df = []
		list_of_labels = []
		for df in list_of_df:
			string_with_text = ""
			for row in df:
				stems = str(df['stems'])
				string_with_text+=stems
			label = df['labels'].unique()
			list_of_labels.append(label)
			list_of_stems_df.append(string_with_text)

		gen_docs = [[w for w in word_tokenize(text)] 
            for text in list_of_stems_df]
		dictionary = gensim.corpora.Dictionary(gen_docs)
		corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
		tf_idf = gensim.models.TfidfModel(corpus)
		sims = gensim.similarities.Similarity('.',tf_idf[corpus], num_features=len(dictionary)) # here we create the model to evaluate similarity
	
	# RESTART THE PROCESS FOR THE NEW DATASET
		print(len(data2))
		results = pd.DataFrame()
		for n,article in enumerate(data2['stems']):
			article = str(article) # this line is to avoid the nan values
			query_doc = [w for w in word_tokenize(article)]
			query_doc_bow = dictionary.doc2bow(query_doc) # doc2vow converts our corpus to bag-of-words
			query_doc_tf_idf = tf_idf[query_doc_bow]
			sims_list = list(sims[query_doc_tf_idf])
			result = pd.DataFrame({'labels_new':list_of_labels, 'similarity': sims_list, 'DIRECCION':data2.iloc[n]['DIRECCION']})
			results = pd.concat([results,result])

		results.labels_new = results.labels_new.map(int)
		results = results[results['similarity']>threshold]
		grouped = results.groupby(['labels_new'])
		results = grouped.filter(lambda x: x['DIRECCION'].count() > min_articles) # checks that more than 7 news in the new month have been detected

		new = results.merge(data2, how = 'inner', on = 'DIRECCION')
		new = new.drop('labels', axis = 1)
		new.rename(columns={'labels_new':'labels'}, inplace=True)
		print(len(new))
		new2 = new.merge(df1, on = 'labels')

		final = pd.concat([final, new2], ignore_index=True, sort = False)
		print(len(final))

		if len(results) < min_articles:
			break

finalized_df = final

print("Finished week " + starting +"!")

# NOW WE START IN THE SECOND WEEK'S RESULTS
while b < n_weeks:
	b = b + 1
	counter = "%02d" % (int(b))
	starting = a + str(counter)
	print(starting)

	c = b - 1
	counter = "%02d" % (int(c))
	following = a + str(counter)
	print(following)

	data1 = pd.read_csv("Final/Chosen_clusters"+starting+".csv")
	data2 = pd.read_csv("Results/Divisio"+following+".csv")

	list_of_df = []
	for label in data1['labels'].unique():
		df = data1[data1['labels']==label]
		list_of_df.append(df)

	list_of_stems_df = []
	list_of_labels = []

	for df in list_of_df:
		string_with_text = ""
		for row in df:
			stems = str(df['stems'])
			string_with_text+=stems
		label = df['labels'].unique()
		list_of_labels.append(label)
		list_of_stems_df.append(string_with_text)

	gen_docs = [[w for w in word_tokenize(text)] 
            for text in list_of_stems_df]
	dictionary = gensim.corpora.Dictionary(gen_docs)
	corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
	tf_idf = gensim.models.TfidfModel(corpus)
	sims = gensim.similarities.Similarity('.',tf_idf[corpus], num_features=len(dictionary)) # here we create the model to evaluate similarity

# COMPAREM AMB TOTS ELS ARTICLES DE WEEK2
	print(len(data2))
	results = pd.DataFrame()
	for n,article in enumerate(data2['stems']):
		article = str(article) # this line is to avoid the nan values
		query_doc = [w for w in word_tokenize(article)]
		query_doc_bow = dictionary.doc2bow(query_doc) # doc2vow converts our corpus to bag-of-words
		query_doc_tf_idf = tf_idf[query_doc_bow]
		sims_list = list(sims[query_doc_tf_idf])
		result = pd.DataFrame({'labels_new':list_of_labels, 'similarity': sims_list, 'DIRECCION':data2.iloc[n]['DIRECCION']})
		results = pd.concat([results,result])

	results.labels_new = results.labels_new.map(int)
	results = results[results['similarity']>threshold]
	grouped = results.groupby(['labels_new'])
	results = grouped.filter(lambda x: x['DIRECCION'].count() > min_articles) # checks that more than 7 news in the new month have been detected

	new = results.merge(data2, how = 'inner', on = 'DIRECCION')
	new = new.drop('labels', axis = 1)
	new.rename(columns={'labels_new':'labels'}, inplace=True)
	print(len(new))
	new2 = pd.concat([data1, new], ignore_index=True, sort = False)
	print(len(new2))

	names = new2['labels'].unique()
	numbers = range(int(max(final['final_labels']))+1, int(max(final['final_labels'])) + len(names)+1)
	df1 = pd.DataFrame({'labels':names, 'final_labels': numbers})
	print(df1)
	# merge those numbers to the df calling themm final_labels
	final = new2.merge(df1, on = 'labels') # new final, previous one gets deteled (we saved it as final1)

	results2 = results
	if len(results2) >= min_articles:
		while c > 0: # when we have the data this should be 52
			c = c - 1
			counter = "%02d" % (int(c))
			following = a + str(counter)
			print(following)
			data2 = pd.read_csv("Results/Divisio"+following+".csv")
			continuing_labels = results2['labels_new'].unique()
			data1 = data1.loc[data1['labels'].isin(continuing_labels)] 
			print(data1['labels'].unique())
	# CREATE A NEW MODEL WITH THE RAMAINING LABELS
			list_of_df = []
			for label in data1['labels'].unique():
				df = data1[data1['labels']==label]
				list_of_df.append(df)
			list_of_stems_df = []
			list_of_labels = []
			for df in list_of_df:
				string_with_text = ""
				for row in df:
					stems = str(df['stems'])
					string_with_text+=stems
				label = df['labels'].unique()
				list_of_labels.append(label)
				list_of_stems_df.append(string_with_text)

			gen_docs = [[w for w in word_tokenize(text)] 
            for text in list_of_stems_df]
			dictionary = gensim.corpora.Dictionary(gen_docs)
			corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
			tf_idf = gensim.models.TfidfModel(corpus)
			sims = gensim.similarities.Similarity('.',tf_idf[corpus], num_features=len(dictionary)) # here we create the model to evaluate similarity
	
	# RESTART THE PROCESS FOR THE NEW DATASET
			print(len(data2))
			results2 = pd.DataFrame()
			for n,article in enumerate(data2['stems']):
				article = str(article) # this line is to avoid the nan values
				query_doc = [w for w in word_tokenize(article)]
				query_doc_bow = dictionary.doc2bow(query_doc) # doc2vow converts our corpus to bag-of-words
				query_doc_tf_idf = tf_idf[query_doc_bow]
				sims_list = list(sims[query_doc_tf_idf])
				result = pd.DataFrame({'labels_new':list_of_labels, 'similarity': sims_list, 'DIRECCION':data2.iloc[n]['DIRECCION']})
				results2 = pd.concat([results2,result])

			results2.labels_new = results2.labels_new.map(int)
			results2 = results2[results2['similarity']>threshold]
			grouped = results2.groupby(['labels_new'])
			results2 = grouped.filter(lambda x: x['DIRECCION'].count() > min_articles) # checks that more than 7 news in the new month have been detected

			new = results2.merge(data2, how = 'inner', on = 'DIRECCION')
			new = new.drop('labels', axis = 1)
			new.rename(columns={'labels_new':'labels'}, inplace=True)
			print(len(new))
			new2 = new.merge(df1, on = 'labels')

			final = pd.concat([final, new2], ignore_index=True, sort = False)
			print(len(final))

			if len(results2) < min_articles:
				break	

	if len(results) >= min_articles:
		c = b
		while c < n_weeks: # when we have the data this should be 52
			c = c + 1
			counter = "%02d" % (int(c))
			following = a + str(counter)
			print(following)
			data2 = pd.read_csv("Results/Divisio"+following+".csv")
		#print(results.head())
			continuing_labels = results['labels_new'].unique()
			data1 = data1.loc[data1['labels'].isin(continuing_labels)] 
			print(data1['labels'].unique())
	# CREATE A NEW MODEL WITH THE RAMAINING LABELS
			list_of_df = []
			for label in data1['labels'].unique():
				df = data1[data1['labels']==label]
				list_of_df.append(df)
			list_of_stems_df = []
			list_of_labels = []
			for df in list_of_df:
				string_with_text = ""
				for row in df:
					stems = str(df['stems'])
					string_with_text+=stems
				label = df['labels'].unique()
				list_of_labels.append(label)
				list_of_stems_df.append(string_with_text)

			gen_docs = [[w for w in word_tokenize(text)] 
            for text in list_of_stems_df]
			dictionary = gensim.corpora.Dictionary(gen_docs)
			corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
			tf_idf = gensim.models.TfidfModel(corpus)
			sims = gensim.similarities.Similarity('.',tf_idf[corpus], num_features=len(dictionary)) # here we create the model to evaluate similarity
	
	# RESTART THE PROCESS FOR THE NEW WEEK
			print(len(data2))
			results = pd.DataFrame()
			for n,article in enumerate(data2['stems']):
				article = str(article) # this line is to avoid the nan values
				query_doc = [w for w in word_tokenize(article)]
				query_doc_bow = dictionary.doc2bow(query_doc) # doc2vow converts our corpus to bag-of-words
				query_doc_tf_idf = tf_idf[query_doc_bow]
				sims_list = list(sims[query_doc_tf_idf])
				result = pd.DataFrame({'labels_new':list_of_labels, 'similarity': sims_list, 'DIRECCION':data2.iloc[n]['DIRECCION']})
				results = pd.concat([results,result])

			results.labels_new = results.labels_new.map(int)
			results = results[results['similarity']>threshold]
			grouped = results.groupby(['labels_new'])
			results = grouped.filter(lambda x: x['DIRECCION'].count() > min_articles) # checks that more than 7 news in the new month have been detected

			new = results.merge(data2, how = 'inner', on = 'DIRECCION')
			new = new.drop('labels', axis = 1)
			new.rename(columns={'labels_new':'labels'}, inplace=True)
			print(len(new))
			new2 = new.merge(df1, on = 'labels')

			final = pd.concat([final, new2], ignore_index=True, sort = False)
			print(len(final))

			if len(results) < min_articles:
				break

	# MERGE CLUSTERS 30% OF WHOSE ARTICLES ARE THE SAME
	for i in finalized_df['final_labels'].unique():
		previous = finalized_df[finalized_df['final_labels'] == i]
		for d in final['final_labels'].unique():
			now = final[final['final_labels'] == d]
			s1 = pd.merge(previous, now, how='inner', on=['DIRECCION'])
			value = len(s1)/len(now) # calculates the percentage of articles in the new clusters that are already in a previous cluster
			print(value)
			value2 = len(s1)/len(previous) # calucaltes the percentage of articles in the new clusters that are already in a previous cluster
			print(value2)
			if value > fusion_threshold: # if it's bigger than 0.4 then the articles in the new cluster get the label of the previous cluster, they become the same
				print('1.Label ' + str(d) + ' becomes label ' + str(i))
				final['final_labels'] = final['final_labels'].replace(d, i)
			elif value2 > fusion_threshold:
				print('2.Label ' + str(i) + ' becomes label ' + str(d))
				finalized_df['final_labels'] = finalized_df['final_labels'].replace(i, d)				
			# afegir linea per si passa al revés!

	finalized_df = pd.concat([finalized_df, final])
	print("Finished week " + starting +"!")

print(finalized_df.groupby(['final_labels'])['DIRECCION'].size())
print(finalized_df.head())

finalized_df = finalized_df.drop_duplicates(subset = ["DIRECCION", "final_labels"])

finalized_df.to_csv('b-final_links3.csv', index = False)



