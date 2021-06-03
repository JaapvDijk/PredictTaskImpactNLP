import random
import numpy as np
import pandas as pd

from classes import data_manager
from classes import embedding_models
from classes import prediction_models
from classes import feature_engineering
from classes import plots
from classes import utils
from classes.embedding_models import SavedEmbeddings
from classes.embedding_models import S2B_Embeddings

seed = 42
random.seed(seed)
np.random.seed(seed)

# data_manager.create_file_with_issues(domain='https://labela.atlassian.net', name='issues11')

df = utils.xlsx_to_dataframe(file='data/cleaned_issues10.xlsx')

# df = feature_engineering.apply_all(df)

# df = data_manager.clean_dataframe(df, filter_on_description=True)

# df = data_manager.get_balanced_data(df, target='story_points')

# text = data_manager.get_merged_text_series(df, 'title', 'description')

# embedding_models.create_s2b_embeddings(text,
#                                        S2B_Embeddings.PARAPHRASE_LANG.value,
#                                        save=True,
#                                        filename='paraphrase_v3')
df['text_embedding'] = embedding_models.get_embedding(SavedEmbeddings.PARAPHRASE_V10_CLASS.value) #'PARAPHRASE_V10.pkl'

# df = df.sample(3000)

# data_manager.get_storypoint_labels(df['story_points'])

X_train, X_test, y_train, y_test = data_manager.get_train_and_test_data(X=df,
                                                                        y=df['story_points'])
# te be removed: data is now scaled in gridsearch pipeline
# X_train, X_test, scaler_X = data_manager.get_scaled_data(X_train,
#                                                          X_test)

train_embeddings = X_train['text_embedding'].to_numpy()
X_input_train = np.column_stack([
                                 X_train['sentiment'].to_numpy(), 
                                 X_train['word_count'].to_numpy(), 
                                 X_train['char_count'].to_numpy(), 
                                 X_train['unique_vs_count'].to_numpy(), 
                                 X_train['unique_w_count'].to_numpy(), 
                                 X_train['avg_word_length'].to_numpy(),
                                 data_manager.squeeze_embeddings(train_embeddings)
                                 ])

test_embeddings = X_test['text_embedding'].to_numpy()
X_input_test = np.column_stack([
                                X_test['sentiment'].to_numpy(),
                                X_test['word_count'].to_numpy(),
                                X_test['char_count'].to_numpy(),
                                X_test['unique_vs_count'].to_numpy(),
                                X_test['unique_w_count'].to_numpy(),
                                X_test['avg_word_length'].to_numpy(),
                                data_manager.squeeze_embeddings(test_embeddings)
                                ])

new_X = np.concatenate((X_input_train, X_input_test), axis=0)
new_y = np.concatenate((y_train, y_test), axis=0)

model = prediction_models.classifiers.create_stacking(new_X, new_y)

# plots.show_confusion_matrix(model, X_input_test, y_test)
# plots.show_loss_plot(model.loss_curve)
# plots.show_shap(model, X_input_train)
# plots.show_learning_curve(model, new_X, new_y)

y_pred = model.predict(X_input_test)
# data_manager.get_regression_scores(y_true=y_test, y_pred=y_pred)
data_manager.get_classification_scores(y_true=y_test, y_pred=y_pred)
