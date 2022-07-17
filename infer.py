from model import Model
from sklearn.feature_extraction.text import TfidfVectorizer


class Inference(Model):
    def __init__(self):
        super().__init__()

    # function infers predictions from trained data
    def predict(self, data_frame, classifier, data: list):
        vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
        vectorizer.fit_transform(data_frame)  # irrelevant, can't transform without fit_transform
        data = vectorizer.transform(data)
        prediction = classifier.predict(data)
        if prediction[0] == 0:
            return 'Negative'
        else:
            return 'Positive'
