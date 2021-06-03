import sys
sys.path.append('../classes')

import embedding_models
import data_manager

import random
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.multioutput import MultiOutputClassifier
from hpsklearn import HyperoptEstimator, any_sparse_classifier, tfidf
from sklearn import metrics
from hyperopt import tpe
import tensorflow as tf
import autokeras as ak
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# mse:3.3277179614737147
# cmape:0.3620654
# mae:1.2641506091710835
# LOSS: MAE - word_averaging (2000 epochs, 100 trials)

# mse:3.213276608978521
# cmape: 0.35828668
# mae:1.2391918447207035
# LOSS: MAE paraphrase3 (2000 epochs, 100 trials)

# mse:3.1151825882658533
# cmape:0.36532894
# mae:1.2457031660868065
# LOSS: MAE paraphrase4(2000 epochs, 100 trials)

# mse:3.249929450725324
# cmape:0.369203

# mae:1.275
# mse:3.4
# LOSS: MAE roberta4(2000 epochs, 100 trials)

# rmse:1.695
# LOSS: RMSE paraphase(2000 epochs, 100 trials)

# rmse: -
# LOSS: RMSE paraphase3(2000 epochs, 100 trials)

df = data_manager.file_to_dataframe(file='../data/cleaned_issues3.xlsx') #.head(2000)

# text = df['title'].to_numpy() + ' ' + df['description'].to_numpy()
storypoints = df['story_points'].to_numpy()

embeddings = embedding_models.get_roberta3()
# embeddings = embedding_models.create_word_averaging(df['title'])

X = embeddings
y = storypoints

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, mode='min') #0001
]

X_train, X_test, y_train, y_test, scaler_X, scaler_y = data_manager.get_scaled_train_and_test_data(X=embeddings, y=y, transform_X=True, transform_y=False)

reg = ak.StructuredDataRegressor(loss='mae', metrics=['mae', 'mse'], overwrite=True, max_trials=50, seed=42, output_dim=1)
reg.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test) , callbacks=callbacks)

y_pred = reg.predict(X_test)
reg.evaluate(X_test, y_test)

print('y_pred: ', y_pred[:20])
print('y_true: ', y_test[:20])

model = reg.export_model()
model.save("test", save_format="tf")
model.save("test.h5")

loaded_model = load_model("test", custom_objects=ak.CUSTOM_OBJECTS)
y_pred = loaded_model.predict(X_test)
data_manager.get_regression_scores(y_true=y_test, y_pred=y_pred)