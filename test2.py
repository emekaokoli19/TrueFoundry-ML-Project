from test import Model
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


class Inference(Model):
    def __init__(self):
        super().__init__()
        pass

    def predict(self, data_frame, classifier, data: list):
        vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
        vectorizer.fit_transform(data_frame)  # irrelevant, can't transform without fit_transform
        data = vectorizer.transform(data)
        prediction = classifier.predict(data)
        if prediction[0] == 0:
            return 'Negative'
        else:
            return 'Positive'


test = Inference()
acc, df_x, clf = test.train(0.2)
test.predict(df_x, clf, ["I love a beautiful day"])
