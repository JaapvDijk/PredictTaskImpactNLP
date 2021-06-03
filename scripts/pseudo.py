import sys
sys.path.append('../classes')

from classes import data_manager
from classes import embedding_models

from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error


def cosine_scores(q, embeddings):
    res = []
    for e in embeddings:
        res.add(dot(e, q)/(norm(e)*norm(q)))
    return res


embeddings = get_embeddings(texts)

predictions = []
for query in embeddings:
    top_5_cosine_scores = cosine_scores(query, embeddings).top(5)

    prediction = mean(storypoints[top_5_indexes])
    predictions.add(prediction)

embedding_performance = get_error_scores(storypoints, predictions)

