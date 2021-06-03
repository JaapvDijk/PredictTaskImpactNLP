# from bert_serving.client import BertClient
# bc = BertClient()
# a=bc.encode(['First do it', 'then do it right', 'then do it better'])
# print(a)

# import torch
# from transformers import BertTokenizer, BertModel
# import logging
# import matplotlib.pyplot as plt

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# text = "Here is the sentence I want embeddings for."
# marked_text = "[CLS] " + text + " [SEP]"

# tokenized_text = tokenizer.tokenize(marked_text)

# print (tokenized_text)

# model.eval()

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
import matplotlib.pyplot as plt
import random
from keras .initializers import Constant
import sentence_transformers
from scipy import spatial
from sent2vec.vectorizer import Vectorizer
from bert_serving.client import BertClient
from sentence_transformers import SentenceTransformer
import get_data as data_stuff
import pickle
from sentence_transformers import SentenceTransformer, util

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


# df = pd.read_excel('data/issues.xlsx', index_col=0)  # .head(5000) 
df = pd.read_excel('data/preprocessed_issues.xlsx', index_col=0)

print("data loaded n: " + str(len(df)))

# df = data_stuff.do_preprocessing(df)
# df.to_excel("data/preprocessed_issues.xlsx")

print("number of issues after prepro: " + str(len(df)))

# model = SentenceTransformer('stsb-roberta-base')

# sentences = df['issue_desc'].to_numpy()
# sentence_embeddings = model.encode(sentences)

infile = open('sentence_embeddings/stsb-roberta-base.pkl','rb')
sentence_embeddings = pickle.load(infile)
infile.close()

X = sentence_embeddings
y = df['velocity']

# outfile = open('sentence_embeddings/--','wb')
# pickle.dump(sentence_embeddings,outfile)
# outfile.close()

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y,
                                                    test_size=0.25,
                                                    random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("got test and train data")

clf = Sequential()
clf.add(Dense(100, input_dim=768, kernel_initializer='normal', activation='relu'))
clf.add(Dense(20, activation='relu'))
clf.add(Dense(10, activation='relu'))
clf.add(Dense(1, activation='linear'))

clf.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=['mse','mae'])
history = clf.fit(X_train, y_train, epochs=12, batch_size=50, verbose=0, validation_split=0.2)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

print("done training network")

#############################
y_pred = np.asarray(clf.predict(X_test))
y_true = np.asarray(y_test)
print('y_pred: ' + str(y_pred[40:60].round(1)))
print('y_true: ' + str(y_true[40:60]))

tot = 0
for i in range(len(y_pred)):
    diff = abs(y_pred[i] - y_true[i])
    tot = tot + diff

print("n issues prepro: " + str(len(X)))
print("totaal afwijking: " + str(tot)  + " total true: " + str(sum(y_true)) + " n: " + str(len(y_true)) + " avg y_true: " + str(sum(y_true) / len(y_true)) + " gem afwijking: " + str(tot/len(y_true)))

# print(accuracy_score(y_true, y_pred))
print("mape: " + str(mean_absolute_percentage_error(y_true, y_pred)))
print("mse: " + str(mean_squared_error(y_true, y_pred)))