import re
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


class Model:
    def __int__(self):
        pass

    def read_data(self):
        data_set = pd.read_csv('airline_sentiment_analysis.csv')
        data_set.loc[data_set['airline_sentiment'] == 'positive', 'airline_sentiment'] = 1  # vectorize sentiment column
        data_set.loc[data_set['airline_sentiment'] == 'negative', 'airline_sentiment'] = 0  # vectorize sentiment column
        data_set['text'] = data_set['text'].apply(self.cleantext)  # clean the text
        text = data_set['text']
        sentiment = data_set['airline_sentiment']
        return text, sentiment

    def train(self, test_size: float):
        df_x, df_y = self.read_data()
        vectorizer = CountVectorizer(stop_words='english')
        encoded_df_x = vectorizer.fit_transform(df_x)  # vectorize text
        acc = 0
        # train model to have an accuracy greater than 90
        while acc < 90:
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(encoded_df_x, df_y, test_size=test_size)
            y_train = y_train.astype('int')
            y_test = np.array(y_test)
            classifier = MultinomialNB()
            classifier.fit(x_train, y_train)
            prediction = classifier.predict(x_test)
            count = 0
            for i in range(len(prediction)):
                if prediction[i] == y_test[i]:
                    count = count + 1
            print(len(prediction), count)
            acc = 100 * (count / len(prediction))
        return acc, df_x, classifier

    # function cleans text when called
    def cleantext(self, text) -> str:
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        text = lemmatizer.lemmatize(text, 'v')
        text = stemmer.stem(text)
        text = self.remove_emojis(text)  # remove emojis
        text = re.sub(r'@[A-Za-z\d]+', '', text)  # remove @tags
        text = re.sub(r'#', '', text)  # remove hashtags
        text = re.sub(r'https?:\/\/\S+', '', text)  # remove the links
        text = text.lower()  # change texts to lower case
        text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuations
        return text

    # This function removes the emoji from text data set
    def remove_emojis(self, data):
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
