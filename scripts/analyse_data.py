import sys, os
#sys.path.append(os.path.join(os.path.dirname(__file__), '../classes'))


import random
import pandas as pd
import numpy as np

from pylab import *
from scipy import *
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns

df = pd.read_excel('data/cleaned_issues10.xlsx') #.head(5000)
pd.options.display.max_colwidth = 2000

print(len(df), sum(df['story_points']))

def get_col_count_and_percentage(col_name):
	col = df[col_name]
	count = col.value_counts(dropna=False)
	percentage = col.value_counts(normalize=True, dropna=False).mul(100).round(1).astype(str) + '%'
	return pd.DataFrame({'count': count, '%': percentage})

#https://towardsdatascience.com/clean-your-data-with-unsupervised-machine-learning-8491af733595
def uniqueWords(X):
	X = str(X)
	X = X.split(' ')
	X = set(X)
	X = len(X)
	return X


def column_completeness():
	msno.matrix(df.sample(500))


def seaborn_bubble(df, charCount, wordCount, uniqueWords):
	X = df[[charCount, wordCount, uniqueWords]]
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	kmeans = KMeans(n_clusters=7, random_state=0).fit(X)
	df['cluster'] = kmeans.labels_

	# mask0 = (df['cluster'] == 0)
	# df0 = df.loc[mask0]
	# desc0 = str(df0['description'].iloc[2:4].to_string())

	# mask6 = (df['cluster'] == 6)
	# df6 = df.loc[mask6]
	# desc6 = str(df6['description'].iloc[2:4].to_string())

	# mask3 = (df['cluster'] == 3)
	# df3 = df.loc[mask3]
	# desc3 = str(df3['description'].iloc[2:4].to_string())

	# with open('0.txt', 'w') as f:
	# 	f.write(desc0)
	# with open('6.txt', 'w') as f:
	# 	f.write(desc6)
	# with open('3.txt', 'w') as f:
	# 	f.write(desc3)

	sns.scatterplot(data=df,
					x='wordCount',
					y='uniqueWords',
					hue='cluster',
					size='charCount',
					legend=True,
					alpha=0.4,
					sizes=(10, 500),
					palette='viridis',
					edgecolors='black')
	plt.show()


def add_title_feature_counts(df, col):
	df['title_charCount'] = df['title'].str.len()
	df['title_wordCount'] = df['title'].str.split(' ').str.len()
	df['title_uniqueWords'] = df['title'].map(lambda title: uniqueWords(title))

	df['description_charCount'] = df['description'].str.len()
	df['description_wordCount'] = df['description'].str.split(' ').str.len()
	df['description_uniqueWords'] = df['description'].map(lambda title: uniqueWords(title))

	return df

# df = add_feature_counts(df, 'title')
# seaborn_bubble(df, 'title_charCount', 'title_wordCount', 'title_uniqueWords')
# df.fillna('',inplace=True)
# =
# seaborn_bubble(df, 'description_charCount', 'description_wordCount', 'description_uniqueWords')

# column_info()
# print_col_count_and_percentage('description')

print(get_col_count_and_percentage('story_points')[:10])