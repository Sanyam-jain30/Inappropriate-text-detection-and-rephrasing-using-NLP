import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import unidecode
import warnings
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob
from transformers import *

warnings.filterwarnings('ignore')

model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")

bad_words = pd.read_csv('bad-words.csv')
df = pd.read_csv('train.csv')

df.columns

df['compound'] = df.sum(axis = 1)

def a(x) :
    if x> 0:
        return 1
    else:
        return 0

df['class'] = df['compound'].apply(lambda x: a(x))
df.head()

df = df[['comment_text', 'class']]
df.head()

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

def getVectorized(arr_feed):
    return vectorizer.transform(arr_feed).toarray()

# LSA (Latent semantic analysis)
from time import time
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
t0 = time()
data = lsa.fit_transform(data)
explained_variance = lsa[0].explained_variance_ratio_.sum()

print(f"LSA done in {time() - t0:.3f} s")
print(f"Explained variance of the SVD step: {explained_variance * 100:.1f}%")

def getLsa(y):
    return lsa.transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, df['class'], test_size=0.3, random_state=0)

acc_dic = {}

# Fitting Decision tree classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
dt.fit(X_train, y_train)

# Predicting the Test set results
y_pred = dt.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score

print('Decision tree classifier: ')

cm = confusion_matrix(y_test, y_pred)
print(cm)

acc = accuracy_score(y_test, y_pred)
print('Accuracy: %.3f' % acc)
acc_dic['Decision tree classifier'] = acc


# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=0)
lr.fit(X_train, y_train)

# Predicting the Test set results
y_pred = lr.predict(X_test)

print('Logistic regression: ')

cm = confusion_matrix(y_test, y_pred)
print(cm)

acc = accuracy_score(y_test, y_pred)
print('Accuracy: %.3f' % acc)
acc_dic['Logistic regression'] = acc


# Fitting Naive bayes classifier to the Training set
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)

# Predicting the Test set results
y_pred = nb.predict(X_test)

print('Naive Bayes classifier: ')

cm = confusion_matrix(y_test, y_pred)
print(cm)

acc = accuracy_score(y_test, y_pred)
print('Accuracy: %.3f' % acc)
acc_dic['Naive Bayes classifier'] = acc


# Fitting Random Forest classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
rf.fit(X_train, y_train)

# Predicting the Test set results
y_pred = rf.predict(X_test)

print('Random forest classifier: ')

cm = confusion_matrix(y_test, y_pred)
print(cm)

acc = accuracy_score(y_test, y_pred)
print('Accuracy: %.3f' % acc)
acc_dic['Random forest classifier'] = acc


# Fitting KNN classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2)
knn.fit(X_train, y_train)

# Predicting the Test set results
y_pred = knn.predict(X_test)

print('KNN classifier: ')
 
cm = confusion_matrix(y_test, y_pred)
print(cm)

acc = accuracy_score(y_test, y_pred)
print('Accuracy: %.3f' % acc)
acc_dic['KNN classifier'] = acc


# Fitting Support vector classifier to the Training set
from sklearn.svm import SVC
svc = SVC(kernel='rbf', random_state=0)
svc.fit(X_train, y_train)

# Predicting the Test set results
y_pred = svc.predict(X_test)

print('Support vector classifier: ')
 
cm = confusion_matrix(y_test, y_pred)
print(cm)

acc = accuracy_score(y_test, y_pred)
print('Accuracy: %.3f' % acc)
acc_dic['Support vector classifier'] = acc


# Training the XGB Classifier model on the Training set
from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators = 1000, learning_rate = 0.1, max_depth = 3)
xgb.fit(X_train, y_train)

# Predicting the Test set results
y_pred = xgb.predict(X_test)

print('XGB classifier: ')
 
cm = confusion_matrix(y_test, y_pred)
print(cm)

acc = accuracy_score(y_test, y_pred)
print('Accuracy: %.3f' % acc)
acc_dic['XGB classifier'] = acc


import pickle
# save the model to disk
filename = 'my-model.h5'

# Training the SGD Classifier model on the Training set
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(max_iter = 1000, penalty = "elasticnet")
sgd.fit(X_train, y_train)
pickle.dump(sgd, open(filename, 'wb'))

# Predicting the Test set results
y_pred = sgd.predict(X_test)

print('SGD classifier: ')
 
cm = confusion_matrix(y_test, y_pred)
print(cm)

acc = accuracy_score(y_test, y_pred)
print('Accuracy: %.3f' % acc)
acc_dic['SGD classifier'] = acc

acc_dic = sorted(acc_dic.items(), key=lambda x:x[1])
print("The best model is: ", next(reversed(acc_dic)))

model_name = next(reversed(acc_dic))[0]
print(model_name)


prev = input('Enter the message: ')

feed= expand_contractions(prev)

feed = feed.lower()

feed = re.sub('[%s]' % re.escape(string.punctuation), '' , feed)

feed = re.sub('\W+',' ', feed)

feed = unidecode.unidecode(feed)

arr_feed = [feed]
arr_feed = vectorizer.transform(arr_feed).toarray()
y = pd.DataFrame(data=arr_feed, columns=vectorizer.get_feature_names_out())
y = lsa.transform(y)

result = sgd.predict(y)
print('Inappropriate senetence' if(result[0] == 1) else 'Appropriate senetence')

def correct_sentence_spelling(sentence):
    sentence = TextBlob(sentence)
    result = sentence.correct()    
    return result

def get_paraphrased_sentences(model, tokenizer, sentence, num_return_sequences=5, num_beams=5):
  inputs = tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt")
  outputs = model.generate(
    **inputs,
    num_beams=num_beams,
    num_return_sequences=num_return_sequences,
  )
  return tokenizer.batch_decode(outputs, skip_special_tokens=True)

if(result[0] == 1):
    ps = PorterStemmer()
    content = [ps.stem(word) for word in feed.split(' ') if not word in set(list(bad_words['words']))]
    content = ' '.join(content)
    
    content = correct_sentence_spelling(content)
    sentencelist = get_paraphrased_sentences(model, tokenizer, str(content), num_beams=10, num_return_sequences=10)
    content = sentencelist[-1]
    
    print(content)
else:
    print(prev)