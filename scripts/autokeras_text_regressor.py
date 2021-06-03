import sys
sys.path.append('../classes')
import word2vec
sys.path.append('../helpers')
import data_helper

import random
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from hpsklearn import HyperoptEstimator, any_sparse_classifier, tfidf
from sklearn import metrics
from hyperopt import tpe
import tensorflow as tf
import autokeras as ak
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

df = pd.read_excel('../../data/cleaned_issues3.xlsx', index_col=0)
print('total ', len(df))    

# desc_split = [str(desc).split() for desc in df['title']]
y = df['story_points'].values
X = df['title'].values

X_train, X_test, y_train, y_test = train_test_split(X, 
                                            y,
                                            test_size=0.25,
                                            random_state=42)

reg = ak.TextRegressor(overwrite=True, max_trials=100, loss=rmse, metrics=['mse', 'mae'])
reg.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test))
y_pred = reg.predict(X_test)

print('ev: ', reg.evaluate(X_test, y_test))
print('y_pred: ', y_pred[:20])
print('y_true: ', y_test[:20])

print('cmape: ', custom_absolute_percentage_error(y_test, y_pred))
print("mape: ", str(mean_absolute_percentage_error(y_test, y_pred)))
print("mse: ", str(mean_squared_error(y_test, y_pred)))
print("mae: ", str(mean_absolute_error(y_test, y_pred)))

model = reg.export_model()
try:
    model.save("text_regressor", save_format="tf")
except Exception:
    model.save("text_regressor.h5")
