import sys, os
sys.path.append('../../classes')

import data_manager
import embedding_models
import prediction_models

import pandas as pd
import numpy as np
import shutil

df = pd.read_excel('../../data/cleaned_issues3.xlsx') #.head(5000) n_rows=10
# df = data_manager.file_to_dataframe(file='data/issues5.xlsx')
# df = df.drop(['issue_id', 'sprint_id', 'board_id', 'state', 'type', 'priority', 'title'], axis=1) #, 'creator', 'assignee' 'description', 'title'


def remove_first_col(df):
    df.drop(df.columns[[0]], axis=1, inplace=True)
    return df


def embedding_as_columns(df):
    df = df.drop(['title'], axis=1)

    #select version corresponding to dataset version
    embeddings = embedding_models.get_paraphrase3()
    embedding_array = []

    for tensor in embeddings:
        embedding_array.append(tensor.numpy())

    #assign every value in the list to its own column in the dataframe
    for i in range(len(embedding_array[0])):
        index_arr = [embedding[i] for embedding in embedding_array]
        df['col'+str(i)] = index_arr

    return df

def fix_storypoints():
    def check(x):
        if x == 'none':
            return None
        else:
            x = str(x).replace(' ', '.')
            return float(x)
    df['story_points'] = df['story_points'].apply(lambda x : check(x))
# fix_storypoints()


def fix_none_all_columns():
    def check(x):
        if x == 'none':
            return None
        else:
            return x

    df = df.applymap(lambda x : check(x))



def create_label():
    for i in range(len(df)):
        score = str(float(df['story_points'].iloc[i]))
        score2 = score.replace('.', '')
        df['story_points'].iloc[i] = score2


def to_zip_labels():
    def assure_path_exists(path):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)


    for i in range(len(df)):
        score = str(float(df['story_points'].iloc[i]))
        score2 = score.replace('.', '')
        path = 'data/'+str(score2)+'/'

        assure_path_exists(path)

        with open(path+'text'+str(i)+'.txt', 'x') as f:
            f.write(df['title'].iloc[i])

    shutil.make_archive("data", "zip", "data")

df = remove_first_col(df)
# df = embedding_as_columns(df)

df.to_csv("../../cleaned_issues3.csv", index=False)
df.to_excel("../../cleaned_issues3.xlsx", index=False)