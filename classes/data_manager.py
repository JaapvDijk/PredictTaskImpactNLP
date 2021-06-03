import sys, os

import pandas as pd
import numpy as np
import re
import pickle
import requests
import json

from classes import utils

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras import backend as K

from gensim.parsing.preprocessing import strip_tags
from gensim.parsing.preprocessing import strip_punctuation 
from gensim.parsing.preprocessing import strip_multiple_whitespaces 
from gensim.parsing.preprocessing import strip_numeric 
from gensim.parsing.preprocessing import strip_short 
from gensim.parsing.preprocessing import stem_text 

from nltk.corpus import stopwords


def get_json_data(url,
                  params = {},
                  mail='jaap.van.dijk@labela.nl',
                  token='gy6T3GIXQw97JhiecVkC7B6F'):
    """
    Returns the json result for the specified (Jira) url.
    Params are required for pagination.
    """
    response = requests.request(
        'GET',
        url,
        headers={'Accept': 'application/json'},
        params=params,
        auth=(mail, token)
    )

    data = json.loads(response.text)
    return data


def get_issues_api_v2(domain):
    """
    Returns Pandas dataframe with all available Jira issues.    
    Retrieves all issues from the specified Jira domain 
    using the /rest/api/2/search?jql= endpoint.

    Can only retrieve issues the authenticated user is allowed to retrieve.
    """

    issues_url = domain + '/rest/api/2/search?jql='
    issues = get_json_data(url=issues_url, params={})

    issue_df = create_initial_issue_dataframe()
 
    print('retrieving issues from: ', issues_url)
    print('max issues: ', issues['maxResults'])

    for i in range(issues['startAt'],
                   issues['total'] + issues['maxResults'],
                   issues['maxResults']):

        issues = get_json_data(issues_url, {'startAt': str(i)})

        if 'issues' in issues:
            for issue in issues['issues']:
                title = utils.deep_get(issue, ['fields','summary'])
                state = utils.deep_get(issue, ['fields','customfield_10500', 0, 'state'])
                priority = utils.deep_get(issue, ['fields','priority', 'name'])
                assignee = utils.deep_get(issue, ['fields','assignee', 'displayName'])
                creator = utils.deep_get(issue, ['fields','creator', 'displayName'])
                description = utils.deep_get(issue, ['fields','description'])
                
                values = {'title': _remove_special_characters(None, title),
                          'assignee': assignee,
                          'creator': creator,
                          'description': _remove_special_characters(None, description),
                          'project': issue['fields']['project']['name'],
                          'story_points': issue['fields']['customfield_10103'],
                          'state': state,
                          'type': issue['fields']['issuetype']['name'],
                          'priority': priority,
                          'board_id': 1,
                          'sprint_id': 1,
                          'issue_id': issue['id']}
                issue_df = issue_df.append(values, ignore_index=True)

                # if len(issue_df) == 200:
                #     print(issue_df['story_points'].to_numpy())
                #     return issue_df

            print('nr issues retrieved: ', str(len(issue_df)))
    return issue_df


def split_texts(text):
    """
    Converts a string/text to an array of strings/words.
    """
    split_text = [str(word).split() for word in text]
    return split_text


def maep_metric(y_true, y_pred):
    """
    Returns the MAE% metric score.
    source: https://towardsdatascience.com/forecast-kpi-rmse-mae-mape-bias-cdc5703d242d
    """
    #some predictions are originaly returned in multiple dimensions. flatten to one.
    if y_pred.shape == (-1,1):
        y_pred = np.reshape(y_pred, (-1, ))

    error_abs = np.absolute(y_pred - y_true)

    sum_error_abs = sum(error_abs)
    sum_y_true = sum(y_true)

    maep = (sum_error_abs / sum_y_true)

    return maep


def rmse_metric(y_true, y_pred):
    """
    Returns the RMSE metric score
    source: https://towardsdatascience.com/forecast-kpi-rmse-mae-mape-bias-cdc5703d242d
    """
    return K.sqrt(K.mean(K.square(y_pred.reshape(-1, ) - y_true))) 


def _remove_stopwords(df, cols):
    """
    Returns a string without words that do not necessarily add meaning to a text
    """
    for col in cols:
        df[col] = df[col].map(lambda text: " ".join([word for word in str(text).split()
                                           if word not in stopwords.words('english')]))
    return df


def _drop_na(df, cols):
    """
    Returns a Pandas dataframe without empty/nan rows in the specified columns.
    """
    df.dropna(subset=cols, thresh=1)
    return df


def _drop_duplicates(df, cols):
    """
    Returns a Pandas dataframe without duplicate rows in the specified columns.
    """
    df.drop_duplicates(subset=cols)
    return df


def _to_float(df, cols):
    """
    Returns a Pandas dataframe where the specified columns are converted to float.
    """
    for col in cols:
        try:
            df[col].astype(float)
        except TypeError:
            print('column cannot be casted to float')
    return df


def _remove_special_characters(df, cols):
    """
    Returns a Pandas dataframe where the specified column has all charaters
    other then 0-9, a-z, A-Z removed, the rest in lowercase.
    """
    def remove_special(text):
        if text is not None:
            return re.sub(r'[^0-9a-zA-Z]+', ' ',  str(text).lower())

    if df is not None:
        for col in cols:
            df[col] = df[col].map(lambda text: remove_special(text))
            return df
    else:
        return remove_special(cols)


def _clean_text(df, cols):
    """
    clean text with gensim:
    - strip_tags
    - strip_punctuation
    - strip_multiple_whitespaces
    - strip_numeric
    - strip_short
    - stem_text
    """
    for col in cols:
        df[col] = df[col].map(lambda text: strip_tags(str(text)))
        df[col] = df[col].map(lambda text: strip_punctuation(str(text)))
        df[col] = df[col].map(lambda text: strip_multiple_whitespaces(str(text)))
        df[col] = df[col].map(lambda text: strip_numeric(str(text)))
        df[col] = df[col].map(lambda text: strip_short(str(text)))
       
    return df


def clean_dataframe(df, filter_on_description=False):
    """
    Returns cleaned dataframe to be used for training and testing (supervised learning).
    Removes rows with empty values (summary, story_points), duplicate rows, certain issue types..
    """
    df = (df.pipe(_drop_na, cols=['title', 'story_points'])
            .pipe(_drop_duplicates, cols=['title', 'story_points'])
            .pipe(_to_float, ['story_points'])
            .pipe(_remove_special_characters, ['title', 'description'])
            .pipe(_clean_text, ['title', 'description'])
            # .pipe(remove_stopwords, ['title', 'description'])
    )

    mask = (df['title'].str.len() > 40) & \
           (df['story_points'] >= 0.5) & \
           (df['story_points'] <= 8) & \
           (df['type'] != 'bug') & \
           (df['type'] != 'epic') & \
           (df['type'] != 'subtaak')
    df = df.loc[mask]

    if filter_on_description and 'description' in df.columns:
        mask = mask & (df['description'].str.len() > 40)

    print("cleaning done, rows left: " + str(len(df)))

    return df


def get_train_and_test_data(X, y):
    """
    Returns the train and testdata to be used for supervised learning.
    Result has 75% train, 25% validation data.
    """
    return train_test_split(X, y, test_size=0.25, random_state=42)


def get_scaled_data(X_train, X_test):
    """
    Returns the dataframe with all numerical features scaled between -1 and 1.
    This exludes features from the blacklist.
    Blacklist columns contain non numerical values, 
    values between -1 and 1, other non scale featur
    """
    scaler_X = None
    scaler_y = None

    #columns/features that should not be scaled
    #TODO
    blacklist = ['board_id', 
                 'sprint_id', 
                 'issue_id', 
                 'assignee',
                 'creator',
                 'description',
                 'state',
                 'type',
                 'priority',
                 'combined_text',
                 'unique_vs_count',
                 'sentiment',
                 'title',
                 'project',
                 'story_points',
                 'Unnamed: 0',
                 'text_embedding']

    for column in X_train:
        if column not in blacklist:
            X_train_data = X_train[column].to_numpy().reshape(-1, 1)
            X_test_data = X_test[column].to_numpy().reshape(-1, 1)

            scaler_X = StandardScaler().fit(X_train_data)
            X_train[column] = scaler_X.transform(X_train_data)
            X_test[column] = scaler_X.transform(X_test_data)   

    # if scale_y:
    #     scaler_y = MinMaxScaler(feature_range=(0, 1)).fit(y_train.reshape(-1, 1))
    #     y_train = scaler_y.transform(y_train.reshape(-1, 1))
    #     y_test = scaler_y.transform(y_test.reshape(-1, 1))

    return X_train, X_test, scaler_X


def squeeze_embeddings(text_embeddings):
    """
    [ [[embedding]], [[embedding]] ] --> [ [embedding], [embedding] ]
    TODO: not being lazy
    """

    new = []
    for embedding in text_embeddings:
        new.append(embedding[0])
    return new


def create_initial_issue_dataframe():
    """
    Returns a Pandas dataframe that has all required 
    columns to store the Jira issue details in.
    """
    issue_df = pd.DataFrame(columns=['title',
                                     'assignee',
                                     'creator',
                                     'description',
                                     'project',
                                     'story_points',
                                     'state',
                                     'type',
                                     'priority',
                                     'board_id',
                                     'sprint_id',
                                     'issue_id'])
    return issue_df


def create_file_with_issues(domain, name):
    """
    Creates a file from all retrieved issues.
    Has .xlsx extension by default, cannot be specified.
    """
    issue_df = get_issues_api_v2(domain)

    print('got all issues from:', domain)

    filename = name+".xlsx"
    
    issue_df.to_excel("data/"+filename, index=False)

    print('created file: ', filename)


def get_regression_scores(y_true, y_pred):
    """
    Returns all relavant metric scores for the impact predictions.
    """
    mse_score = round(mean_squared_error(y_true, y_pred), 3)
    rmse_score = round(rmse_metric(y_true, y_pred).numpy(), 3)
    mae_score = round(mean_absolute_error(y_true, y_pred), 3)
    maep_score = round(maep_metric(y_true, y_pred), 3)

    print('mse:', mse_score)
    print('rmse:', rmse_score)
    print('mae:', mae_score)
    print('maep:', maep_score)

    return mse_score, mae_score, maep_score, rmse_score


def get_classification_scores(y_true, y_pred):
    """
    Returns all relavant metric scores for the impact predictions.
    Is used for classification, may be removed in the future.
    """
    acc_score = accuracy_score(y_true, y_pred)
    pre_score = round(precision_score(y_true, y_pred, average='macro'), 3)
    rec_score = round(recall_score(y_true, y_pred, average='macro'), 3)
    
    print('acc:', acc_score)
    print('pre:', pre_score)
    print('rec:', rec_score)

    return acc_score, pre_score, rec_score


def print_gridsearch_best_stats(gs):
    """
    Returns the details of the best model resulting from a sklearn gridsearch.
    """
    print('best estim: ', gs.best_estimator_)
    print('best parms: ', gs.best_params_)
    print('best score: ', gs.best_score_)
    print('score test 1: ', gs.cv_results_['split0_test_score'])
    print('score test 2: ', gs.cv_results_['split1_test_score'])
    print('score test 3: ', gs.cv_results_['split2_test_score'])
    print('score test 4: ', gs.cv_results_['split3_test_score'])
    print('score test 5: ', gs.cv_results_['split4_test_score'])


def get_storypoint_labels(y):
    """
    Returns list the labels for all storypoints.
    Used for classification
    """
    y[y.astype(float) < 4] = 0
    y[y.astype(float) >= 4] = 1

    # le = preprocessing.LabelEncoder()
    # le.fit(y)


def get_merged_text_series(df, col1, col2):
    df[col1] = df[col1].astype(str)
    df[col2] = df[col2].astype(str)

    merged = np.array(df[col1] + ' ' + df[col2])
    return merged


def get_balanced_data(df, target):
    df1 = (df[df[target].astype(float) <= 3]).head(2980)
    df2 = (df[df[target].astype(float) >= 4]).head(2980)
    df = df1.append(df2)
    print('balancing done, rows left: ', len(df))
    return df
