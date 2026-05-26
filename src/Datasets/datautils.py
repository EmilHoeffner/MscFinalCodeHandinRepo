import numpy as np

import random

from Constants import constants

import sys 

sys.path.append("..")

from Utils.listutils import list_distinct

'''
    Filter questions of less than 3 tokens. Returns the indexes of questions of 3 or more tokens
'''
def filter(questions, contexts, tokenizer):
    indecies = []

    for i,(question, context) in enumerate(zip(questions, contexts)):
        if num_tokens(question, tokenizer) >= constants.min_question_length() and num_tokens(context, tokenizer) <= constants.max_context_length():
            indecies.append(i)
    
    return indecies

def filtered_indices(questions, contexts, tokenizer):
        filter_indexes = filter(questions, contexts, tokenizer)
        num_filtered_away = len(questions) - len(filter_indexes)
        print("Filtered {}/{} samples".format(num_filtered_away, len(questions)))

        return filter_indexes

'''
    sentence: A string

    Returns: Number of tokens in the sentence
'''
def num_tokens(sentence, tokenizer):
    return len(tokenizer.tokenize(sentence))

def find_length_argsort(contexts):
    lengths = np.array([len(context) for context in contexts])
    return np.argsort(a = lengths)

'''
    Return a list of indices of the samples

    k: questions per contexts. 
    n: number of contexts to sample
'''
def sample_by_contexts(contexts, k, n, N):
    random.seed(constants.random_seed())
    # A dictionary which maps every context to the indices of samples with that context
    dict = {}

    # Initialise dictionary to map every context to indices of samples with that context
    for idx, context in enumerate(contexts):
        if context in list(dict.keys()):
            lst = dict[context]
            lst.append(idx)
            dict[context] = lst
        else:
            dict[context] = [idx]

    keys = list(dict.keys())
    # Prune contexts below the frequency threshold
    for context in keys:
        freq = len(dict[context])

        if freq < k:
            del dict[context]

    candidate_contexts = list(dict.keys())

    # https://www.geeksforgeeks.org/python/python-random-sample-function/
    # Sample from the contexts to use.
    selected_contexts = random.sample(candidate_contexts, n)

    # Sample k indices from the chosen contexts. 
    sample_indices = []

    for c in selected_contexts:
        candidate_indices = dict[c]
        selected_indices = random.sample(candidate_indices, k)
        sample_indices += selected_indices

    # Randomly sample N - n extra contexts to the corpus, so the corpus has N distinct contexts. 
    C_extra = [c for c in contexts if c not in selected_contexts]
    C_extra = list_distinct(C_extra)

    extra_contexts = random.sample(C_extra, N - n)

    return sample_indices, extra_contexts
