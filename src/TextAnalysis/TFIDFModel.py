from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# https://stackoverflow.com/questions/35109424/how-to-make-tf-idf-matrix-dense
class TFIDFModel:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(lowercase = True, stop_words = "english")

    '''
        questions: List of strings
        contexts: List of strings
    '''
    def fit(self, questions, contexts):
        combined = questions + contexts 
        self.vectorizer.fit(combined)

        return self.predict(questions, contexts)
    
    '''
        questions: List of strings
        contexts: List of strings
    '''
    def predict(self, questions, contexts):
        X_context = self.vectorizer.transform(contexts).toarray()
        X_question = self.vectorizer.transform(questions).toarray()
        X_total = np.hstack((X_question, X_context))

        return X_total 
    
    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()