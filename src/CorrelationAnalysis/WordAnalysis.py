
import sys
import os 
from dataloader import DataLoader

import nltk
from nltk.tokenize import word_tokenize

sys.path.append("..")

from TextAnalysis.TextAnalysis import word_cloud
from TextAnalysis.TextAnalysis import word_histogram

from Filehandling.FileHandler import FileHandler

import numpy as np


'''
    scores: List of scores
    Contexts: List of contexts

    Compares word cloud of samples with score = 0 and score = 1
'''
def word_cloud_comparison(scores, questions, contexts, dir_path):
    good_contexts = []
    good_questions = []
    bad_contexts = []
    bad_questions = []

    for (score, question, context) in zip(scores, questions, contexts):
        if score <= 0:
            bad_contexts.append(context)
            bad_questions.append(question)
        elif score == 1:
            good_contexts.append(context)
            good_questions.append(question)

    save_path_good = dir_path + "Context_Histogram_Good.png"
    save_path_bad = dir_path + "Context_Histogram_Bad.png"

    word_histogram(texts = good_contexts, save_path = save_path_good, 
                   title = "Histogram over contexts with score = 1", max_words = 10, tokenizer = word_tokenize)
    word_histogram(texts = bad_contexts, save_path = save_path_bad, 
                   title = "Histogram over contexts with score = 1", max_words = 10, tokenizer = word_tokenize)

    save_path_good = dir_path + "Question_WordCloud_Good.png"
    save_path_bad = dir_path + "Question_WordCloud_Bad.png"

    word_histogram(texts = good_questions, save_path = save_path_good, 
                   title = "Word histogram over questions with score = 1" , max_words = 10, tokenizer = word_tokenize)
    word_histogram(texts = bad_questions, save_path = save_path_bad, 
                   title = "Word histogram cloud over questions with score = 0" , max_words = 10, tokenizer = word_tokenize)


def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # https://www.nielit.gov.in/gorakhpur/sites/default/files/Gorakhpur/ALevel_1_Python_20May.pdf
    num_args = len(sys.argv)

    # https://www.geeksforgeeks.org/nlp/removing-stop-words-nltk-python/
    nltk.download('punkt')
    nltk.download('stopwords')

    if num_args != 2:
        raise Exception("Program must be run with exactly one argument")
    
    data_set_iden = sys.argv[1]

    dl = DataLoader(data_set_iden)
    file_handler = FileHandler()

    questions_train = dl.load_train_feature("question")
    contexts_train = dl.load_train_feature("context")

    dir_path = file_handler.storage_path() + f"/CorrelationAnalysis/{data_set_iden}/Word_Cloud/"
    word_cloud_comparison(dl.load_train_feature("EM"), questions_train, contexts_train, dir_path = dir_path)


if __name__ == "__main__":
    main()
