
from classes import data_manager

import numpy as np
import pickle

from enum import Enum

from gensim.models import Word2Vec
from gensim.models import doc2vec
import gensim.downloader as api

from sentence_transformers import SentenceTransformer, util


embedding_dir = 'embedding_vectors/'


class SavedEmbeddings(Enum):
    """
    Contains the path for every saved embedding.
    Embeddings are saved as a pickle object file.

    Key name format: {EmbeddingMethod_DatasetVersion_Misc}
    - EmbeddingMethod: Used method for embedding the texts.
    - DatasetVersion: Dataset version used to create the embeddings.
    - Misc: Any other specifications on the dataset version
    """
    PARAPHRASE_V3 = 'paraphrase3'
    PARAPHRASE_V4_GENSIM = 'paraphrase4_gensim'
    PARAPHRASE_V5_TD = 'paraphrase5_TD'
    PARAPHRASE_V6_BALANCED = 'paraphrase6_reg_balanced'
    PARAPHRASE_V3_ENGLISH_ONLY = 'paraphrase3_english'
    ROBERTA_V3 = 'roberta3'
    ROBERTA_V4 = 'roberta4'
    ROBERTA_V5 = 'roberta5'
    PARAPHRASE_V7_CLASS = 'paraphrase7_class_6000'
    PARAPHRASE_V10_CLASS = 'Paraphrase10_class'


class S2B_Embeddings(Enum):
    """
    Contains names of pretrained s2b models (sentence_transformers)
    To be used in create_s2b_embeddings()
    More info: https://www.sbert.net/docs/pretrained_models.html
    And more: https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/
    """
    STSB_ROBERTA = 'stsb-roberta-large'
    STSB_ROBERTA_LANG = 'stsb-xlm-r-multilingual'
    PARAPHRASE_MPNET = 'paraphrase-mpnet-base-v2'
    PARAPHRASE_LANG = 'paraphrase-xlm-r-multilingual-v1'


def get_embedding(embedding_name):
    """
    Returns embeddings from the specified pickle object
    """
    infile = open(embedding_dir+embedding_name+'.pkl','rb') #'embedding_vectors/'+
    embeddings = pickle.load(infile)
    infile.close()

    print('embeddings loaded: ', len(embeddings))
    #TODO
    embeddings = embeddings.cpu().detach().numpy()

    numpy_embeddings = []
    for embedding in embeddings:
        numpy_embeddings.append([embedding])

    return numpy_embeddings


#https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/data_augmentation/train_sts_seed_optimization.py
def create_s2b_embeddings(text, embedder, save=False, filename='new_embeddings'):
    embedder = SentenceTransformer(embedder)
    embeddings = embedder.encode(sentences=text, convert_to_tensor=True)

    if save:
        save_embeddings(filename, embeddings)
    return embeddings


def create_word_averaging(texts, save=False, filename='word2vec'):
    """
    Takes a list of texts
    Returns the mean of all the word embedding vectors in a sentence. 
    The average word is the representation of the text.
    """
    split_text = data_manager.split_texts(texts)
    missing_words = []

    # embedding_model = api.load('word2vec-google-news-300')

    embedding_model = Word2Vec(split_text, 
                               size=300, 
                               min_count=1, 
                               sg=1, 
                               seed=42, 
                               workers=6)
    # embedding_model = Word2Vec.load("embedding_models/google+labela.model")
    # embedding_model = Word2Vec.load("embedding_models/google+labela.model_3.model")
    
    # embedding_model.intersect_word2vec_format('embedding_models/GoogleNews-vectors-negative300.bin.gz',
    #                                           lockf=1.0,
    #                                           binary=True)
    # embedding_model.train(split_text, 
    #                       total_examples=embedding_model.corpus_count,
    #                       epochs=embedding_model.epochs)
    # embedding_model.save("google+labela.model_3.model")

    def vectorize(sentence):
        vec = []
        numw = 0
        embedding = []
        for w in sentence:
            try:
                if numw == 0:
                    vec = embedding_model[w]
                else:
                    vec = np.add(vec, embedding_model[w])
                numw += 1
            except KeyError:
                missing_words.append(w)
        try:
            embedding = vec / numw # vec / np.sqrt(vec.dot(vec))
        except Exception:
            print('no embedding found')

        return embedding 

    embeddings = []
    for i in range(len(split_text)):
        vec = vectorize(split_text[i])
        embeddings.append(vec)

    print(len(missing_words), " missing words: ", missing_words)

    if save:
        data_manager.save_embeddings(filename, embeddings)

    return embeddings


def create_doc2vec(texts):
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    text_split = [str(text).split() for text in texts]

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
    model = Doc2Vec(documents, vector_size=300, window=3, min_count=1, workers=4)

    embeddings = []
    for text in text_split:
        embedding = model.infer_vector(text)
        embeddings.append(embedding)

    return embeddings


def save_embeddings(name, embeddings):
    """
    Saves object as pickle file. 
    Only used to save larger embeddings so that they can be reused.
    """
    outfile = open(embedding_dir+name+'.pkl', 'wb')
    pickle.dump(embeddings, outfile)
    outfile.close()

#optional: https://github.com/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb
