import sys
sys.path.append('../classes')

import data_manager
import embedding_models

from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import pickle
from scipy import stats
import copy
from scipy import spatial
from sent2vec.vectorizer import Vectorizer

# s2v - sentence_transformers: stsb-roberta-base
# mse:3.14617790452049
# cmape:0.37698793395644054
# mae:1.3203664976763836
# embedder = SentenceTransformer('stsb-roberta-base')

# s2v - sentence_transformers: stsb-roberta-large
# mse:3.142181844106464
# cmape:0.3757863815619756
# mae:1.3140166877904522
# embedder = SentenceTransformer('stsb-roberta-large')

# s2v - sentence_transformers: stsb-bert-large
# mse:3.187697444021969
# cmape:0.38104801151848955
# mae:1.3326373046049853
# embedder = SentenceTransformer('stsb-bert-large')

# s2v - sentence_transformers: stsb-distilbert-base
# mse:3.114541761723701
# cmape:0.3751702198436169
# mae:1.3038022813688215
# embedder = SentenceTransformer('stsb-distilbert-base')

# s2v - sentence_transformers: stsb-bert-base
# mse:3.14617790452049
# cmape:0.37698793395644054
# mae:1.3203664976763836
# embedder = SentenceTransformer('stsb-bert-base')

# s2v - sentence_transformers: paraphrase-xlm-r-multilingual-v1
# mse:3.0061197560202793
# cmape:0.36867409798724377
# mae:1.2796451204055768
# embedder = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
##data4
# mse:3.1056
# cmape:0.368
# mae:1.303

# s2v - sentence_transformers: paraphrase-distilroberta-base-v1
# mse:3.0770712162019436
# cmape:0.37244539617124045
# mae:1.3019977820025348
# embedder = SentenceTransformer('paraphrase-distilroberta-base-v1')


# s2v - sentence_transformers: nli-bert-large
# mse:3.2702805581960286
# cmape:0.38613520303835186
# mae:1.3562140895648498
# embedder = SentenceTransformer('nli-bert-large')

# s2v - sentence_transformers: nli-distilbert-base
# mse:3.1924026959231093
# cmape:0.37878623821799495
# mae:1.320401880016899
# embedder = SentenceTransformer('nli-distilbert-base')

# s2v - sentence_transformers: nli-roberta-large
# mse:3.22287472539079
# cmape:0.3809797125977973
# mae:1.329522602450359
# embedder = SentenceTransformer('nli-roberta-large')

# s2v - sentence_transformers: nli-bert-large-max-pooling
# mse:3.3175029573299537
# cmape:0.38719807314363747
# mae:1.3650950570342204
# embedder = SentenceTransformer('nli-bert-large-max-pooling')

# s2v - sentence_transformers: nli-bert-large-cls-pooling
# mse: 3.2551173241444866
# cmape:0.3843952675536736
# mae:1.347252323616392
# embedder = SentenceTransformer('nli-bert-large-cls-pooling')

# s2v - sentence_transformers: average_word_embeddings_glove.6B.300d
# mse:3.0888964564850023
# cmape:0.3717619037943738
# mae:1.3039311364596538
# embedder = SentenceTransformer('average_word_embeddings_glove.6B.300d')

#word_averaging
# mse:3.0888964564850023
# cmape:0.3717619037943738
# mae:1.3039311364596538

df = data_manager.file_to_dataframe(file='../data/cleaned_issues3.xlsx') #.head(3000)

X = np.asarray(df['title'])
y = np.asarray(df['story_points'])

embeddings = embedding_models.create_word_averaging(X)
# data_manager.save_embeddings('name', embeddings)

y_pred = np.array([])
top_n = 5
top_k = min(top_n+1, len(X))
i = 0
for query in embeddings:
    i += 1
    print(i)
    most_simular_scores = []

    cos_scores = util.pytorch_cos_sim(query, embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    for idx in top_results[1]:
        y_true = y[idx]
        most_simular_scores.append(y_true)
    
    #index 0 is the same sentence, not a simular sentence
    most_simular_scores.pop(0)
    arr_sum = np.sum(np.asarray(sum(most_simular_scores)))
    total = len(most_simular_scores)
    avg = (arr_sum/total)
    # y_pred.insert(avg)
    y_pred = np.append(y_pred, avg)

data_manager.get_regression_scores(y, y_pred)

#500
# mse:3.7019170000000003
# cmape:0.3621758423652507
# mae:1.48014

#1500
# mse:3.4782899333333335
# cmape:0.36753809051306324
# mae:1.4214466666666667

#2500
# mse:3.1257851600000004
# cmape:0.3721599722787513
# mae:1.318668

#3500
# mse:3.178684857142857
# cmape:0.3592584222799742
# mae: 1.3279885714285715

#4500
# mse:3.2491826666666666
# cmape:0.3704436821247372
# mae:1.3453911111111112

#10500
# mse:3.258685567547098
# cmape:0.3756970273493426
# mae:1.342185932026886