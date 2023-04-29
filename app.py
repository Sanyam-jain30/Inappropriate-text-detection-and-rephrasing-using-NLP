from flask import Flask, request, json
import re, string, unidecode
import pandas as pd
import pickle
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob
from transformers import *

bad_words = pd.read_csv('bad-words.csv')

# load the saved model
with open('my-model.h5', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv('train.csv')
df['compound'] = df.iloc[:, 2:].sum(axis=1)

def a(x) :
    if x> 0:
        return 1
    else:
        return 0

df['class'] = df['compound'].apply(lambda x: a(x))
df = df[['comment_text', 'class']]

contractions_dict = { 
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}

contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))

def expand_contractions(text,contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)

# MODEL_FILE_NAME = 'D:\AppsFolder\PythonProjects\Sanyam AI_ML\AI\Course project\Cloud-Project\model.h5'

app = Flask(__name__)


df['comment_text']= df['comment_text'].apply(lambda x:expand_contractions(x))

df.head()

df['comment_text'] = df['comment_text'].str.lower()

df['comment_text'] = df['comment_text'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '' , x))
df.sample(5)

df['comment_text'] = df['comment_text'].apply(lambda x: re.sub('\W+',' ', x))
df.head()

df = pd.concat([df[df['class'] > 0], df[df['class'] == 0].sample(int(len(df[df['class'] > 0])*1.5))], axis = 0).sample(frac = 1)
df[df['class']>=0].hist()

df.shape

df['comment_text'] = df['comment_text'].apply(lambda x: unidecode.unidecode(x))

# Vectorisation
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['comment_text']).toarray()
data = pd.DataFrame(data=X, columns=vectorizer.get_feature_names_out())

# LSA (Latent semantic analysis)
from time import time
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
t0 = time()
data = lsa.fit_transform(data)
explained_variance = lsa[0].explained_variance_ratio_.sum()

@app.route('/result', methods=['POST'])
def index():    
    # Parse request body for model input 
    prev = request.form['feed']
    
    feed = expand_contractions(prev)

    feed = feed.lower()

    feed = re.sub('[%s]' % re.escape(string.punctuation), '' , feed)

    feed = re.sub('\W+',' ', feed)

    feed = unidecode.unidecode(feed)

    print(feed)

    arr_feed = [feed]
    arr_feed = vectorizer.transform(arr_feed).toarray()
    y = pd.DataFrame(data=arr_feed, columns=vectorizer.get_feature_names_out())
    y = lsa.transform(y)

    result = model.predict(y)
    print(result)
    print('Inappropriate senetence' if(result[0] == 1) else 'Appropriate senetence')

    def correct_sentence_spelling(sentence):
        sentence = TextBlob(sentence)
        result = sentence.correct()    
        return result

    if(result[0] == 1):
        ps = PorterStemmer()
        content = [ps.stem(word) for word in feed.split(' ') if not word in set(list(bad_words['words']))]
        content = ' '.join(content)
        
        content = correct_sentence_spelling(content)
        prediction = str(content)
    else:  
        prediction = prev
    result = {'prediction': prediction}    
   
    return json.dumps(result)

if __name__ == '__main__':    
    # listen on all IPs 
    app.run(debug=False, host='0.0.0.0')