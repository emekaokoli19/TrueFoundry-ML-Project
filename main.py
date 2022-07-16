import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn import svm
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
import re
import string
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

data_set = pd.read_csv('airline_sentiment_analysis.csv')
# vectorize sentiment column
data_set.loc[data_set['airline_sentiment'] == 'positive', 'airline_sentiment'] = 1
data_set.loc[data_set['airline_sentiment'] == 'negative', 'airline_sentiment'] = 0


def cleantext(text):
    # stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    text = lemmatizer.lemmatize(text, 'v')
    text = stemmer.stem(text)
    text = remove_emojis(text)  # remove emojis
    text = re.sub(r'@[A-Za-z\d]+', '', text)  # remove @tags
    text = re.sub(r'#', '', text)  # remove hashtags
    text = re.sub(r'https?:\/\/\S+', '', text)  # remove the links
    text = text.lower()  # change texts to lower case
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuations
    # text = text.split()  # tokenize words
    # # remove common words
    # for word in text:
    #     if word in stop_words:
    #         text.remove(word)
    return text


# function removes emojis
def remove_emojis(data):
    emoji = re.compile("["
                       u"\U0001F600-\U0001F64F"  # emoticons
                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                       u"\U00002500-\U00002BEF"  # chinese char
                       u"\U00002702-\U000027B0"
                       u"\U00002702-\U000027B0"
                       u"\U000024C2-\U0001F251"
                       u"\U0001f926-\U0001f937"
                       u"\U00010000-\U0010ffff"
                       u"\u2640-\u2642"
                       u"\u2600-\u2B55"
                       u"\u200d"
                       u"\u23cf"
                       u"\u23e9"
                       u"\u231a"
                       u"\ufe0f"  # dingbats
                       u"\u3030"
                       "]+", re.UNICODE)
    return re.sub(emoji, '', data)


data_set['text'] = data_set['text'].apply(cleantext)
df_x = data_set['text']
df_y = data_set['airline_sentiment']
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(df_x, df_y, test_size=0.2)
# cv = TfidfVectorizer(min_df=1, stop_words='english')
cv = CountVectorizer(stop_words='english')
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
x_train = x_train.toarray()
y_train = y_train.astype('int')
y_test = np.array(y_test)
clf = MultinomialNB()
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
# acc = metrics.accuracy_score(y_test, pred)
score = clf.score(pred, y_test)
count = 0
for i in range(len(pred)):
    if pred[i] == y_test[i]:
        count = count + 1
print(len(pred), count)
print(count/len(pred))
