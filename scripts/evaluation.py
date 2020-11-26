import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='d-categorised_data2.csv', help='Final dataset')
parser.add_argument('--annotations', type=str, default='c-annotated.csv', help='Annotated data')
parser.add_argument('--no_4', type=bool, default=False, help='Delete cluster 4')

args = parser.parse_args()


data1 = pd.read_csv(args.data)
weeks = ['2018-00','2018-01', '2018-02']
data1 = data1[data1.week.isin(weeks)]

print(len(data1))

data = pd.read_csv(args.annotations)
data = data[data.week.isin(weeks)]
#data = data[data.annotation != 0]

print(len(data))

data_merged = data.merge(data1, on = "DIRECCION", how='left')
data_merged.fillna(0, inplace=True)
data_merged = data_merged[data_merged.annotation != 0]

print(data_merged['annotation'].unique())
print(data_merged['final_labels'].unique())

print(len(data_merged)) # the current system allows one article to be in more than one story, for this reason, the lengh of data_merged and data are not the same

plot_name = 'contingency_matrix'

if args.no_4:
	data_merged = data_merged[data_merged.annotation != 4]
	data_merged = data_merged[data_merged.final_labels != 4]
	plot_name = 'contingency_matrix_no_4'

def purity_score (y_true, y_pred):
	contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
	plt.figure(figsize = (4.8,4))
	sns.heatmap(contingency_matrix, annot=False,cmap="YlGnBu", cbar=False)
	#plt.matshow(contingency_matrix)
	plt.ylabel('True cluster')
	plt.xlabel('Assigned cluster')
	#plt.title('Contingency matrix \n with cluster 4')
	plt.savefig(plot_name)
	#plt.savefig('figure6.tiff', dpi=600, format="tiff")
	plt.show()
	return np.sum(np.amax(contingency_matrix, axis = 0)) / np.sum(contingency_matrix)

score = purity_score(data_merged['annotation'], data_merged['final_labels'])

print('Purity:', score)

nmi = metrics.normalized_mutual_info_score(data_merged['annotation'], data_merged['final_labels'])

print('NMI:', nmi)  




