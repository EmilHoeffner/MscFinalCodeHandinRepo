
# https://www.geeksforgeeks.org/nlp/tokenize-text-using-nltk-python/

import nltk
from nltk.tokenize import word_tokenize

import Levenshtein

class Evaluater:
    def __init__(self):
        nltk.download('punkt')

    def jaccard_similarity(self, truth, pred):
        truth_tokens = word_tokenize(truth)
        pred_tokens = word_tokenize(pred)

        truth_set = set(truth_tokens)
        pred_set = set(pred_tokens)

        intersection = truth_set.intersection(pred_set)
        union = truth_set.union(pred_set)

        jaccard = len(intersection) / len(union)

        return jaccard
    
    def recall(self, truth, pred):
        truth_set = set(word_tokenize(truth))

        if len(truth_set) == 0:
            raise Exception("Truth Answer with zero tokens")

        pred_set = set(word_tokenize(pred))

        recalled_tokens = []
        for token in truth_set:
            if token in pred_set:
                recalled_tokens.append(token)

        if len(recalled_tokens) == 0:
            return 0
        else:
            return len(recalled_tokens) / len(truth_set)
    
    def precision(self, truth, pred):
        pred_set = set(word_tokenize(pred))

        if len(pred_set) == 0:
            return 0
        
        truth_set = set(word_tokenize(truth))

        precision_tokens = []

        for token in pred_set:
            if token in truth_set:
                precision_tokens.append(token) 

        if len(precision_tokens) == 0:
            return 0
        
        return len(precision_tokens) / len(pred_set)
    
    def f1(self, truth, pred):
        recall = self.recall(truth, pred)
        precision = self.precision(truth, pred)

        if recall == 0 and precision == 0:
            return 0
        else:
            return 2 * (precision * recall) / (precision + recall)
    
    def exact_match(self, truth, pred):
        return 1 if truth.strip(" ") == pred.strip(" ") else 0
    
    # https://www.geeksforgeeks.org/python/introduction-to-python-levenshtein-module/
    def edit_distance(self, truth, pred):
        return Levenshtein.distance(truth, pred)









