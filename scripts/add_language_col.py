import pandas as pd
from langdetect import detect

a = 0 
df = pd.read_excel('data/preprocessed_issues.xlsx', index_col=0)


def set_languange(x):
    global a
    a = a + 1
    try:
        lang = detect(x)
        print(str(a) + ":" + str(lang))
        return lang
    except:
        print('error')
        return 'error'


df['languange'] = df['issue_desc'].apply(lambda x: set_languange(x))

df.to_excel("data/preprocessed_issues_languange.xlsx")

