import pandas as pd
from textblob import TextBlob

from classes import data_manager

def apply_all(df):
    df['combined_text'] = df['title']
    df['word_count'] = df['combined_text'].map(lambda text: 
                                               get_word_count(text))

    df['char_count'] = df['combined_text'].map(lambda text: 
                                               get_char_count(text))

    df['avg_word_length'] = df['combined_text'].map(lambda text: 
                                                    avg_word_length(text))

    df['unique_w_count'] = df['combined_text'].map(lambda text: 
                                                   get_unique_word_count(text))

    df['unique_vs_count'] = df['combined_text'].map(lambda text: 
                                                    unique_vs_count(text))

    df['sentiment'] = df['combined_text'].map(lambda text: 
                                              sentiment(text))
    # df['count_punctuation'] = combined.map(lambda text: count_punctuations(text))
    # df['count_capital_words'] = combined.map(lambda text: count_capital_words(text))
    # df['title_char_entities'] = df['title'].map(lambda text: get_entities(text))
    print('added extra features..')
    return df

#https://www.analyticsvidhya.com/blog/2021/04/a-guide-to-feature-engineering-in-nlp/
def get_word_count(text):
    if type(text) == str:
        return len(text.split(' '))


def get_char_count(text):
    if type(text) == str:
        return len(text)


def get_unique_word_count(text):
    if type(text) == str:
        return len(set(text.split()))


def unique_vs_count(text):
    unique_word_count = get_unique_word_count(text)
    word_count = get_word_count(text)
    if type(word_count) == int:
        return unique_word_count/word_count
    

def avg_word_length(text):
    char_count = get_char_count(text)
    word_count = get_word_count(text)
    if type(char_count) == int:
        return get_char_count(text) / get_word_count(text)


def count_punctuations(text):
   if type(text) == str:
        punctuations='!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'
        d=dict()
        for i in punctuations:
            d[str(i)+' count']=text.count(i)
        return d


def count_capital_words(text):
    if type(text) == str:
        return sum(map(str.isupper,text.split()))


def sentiment(text):
    if type(text) == str:
        return TextBlob(text).sentiment.polarity


# def get_entities(text):
#     ner = spacy.load("en_core_web_lg")
#     doc = ner(text)
#     print(doc)

