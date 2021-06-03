import pandas as pd
# from translate import Translator
from googletrans import Translator
import time
from google_trans_new import google_translator  

a = 0 
df = pd.read_excel('data/preprocessed_issues.xlsx', index_col=0)


def translate_comment(x, translator):
    global a
    a = a + 1
    try:
        translation = translator.translate(x,lang_tgt='en')
        print(str(a) + ":" + translation)
        return translation
    except:
        print('fail')
        df.to_excel("data/preprocessed_issues_english.xlsx")

# translator= Translator(to_lang="english")
# translator = Translator()
translator = google_translator()  
# translator = google_translator(url_suffix="hk", timeout=5) 
# gs = goslate.Goslate()

for i in range(len(df['issue_desc'])):
    df['issue_desc'].iloc[i] = translate_comment(df['issue_desc'].iloc[i], translator)
    time.sleep(0.5)

df.to_excel("data/preprocessed_issues_english.xlsx")
