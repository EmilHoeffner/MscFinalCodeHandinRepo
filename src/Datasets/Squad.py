
'''
    Some code has been taken from my Msc Thesis Code repo, with the pre-thesis experiment
'''

import sys

from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer


sys.path.append("..")

from Datasets.datautils import filtered_indices
from Datasets.datautils import find_length_argsort 
from Datasets.datautils import sample_by_contexts
from Constants import constants

from Utils.listutils import list_distinct

class Squad:

    def __init__(self, iden):
        ds = load_dataset("rajpurkar/squad")
        self.iden = iden
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
        self.tokenizer.model_max_length = constants.tokenizer_max_length()

        if iden == "Dummy":
            # Using same samples as the preliminary experiment:
            n_sample = 200
            self.N = n_sample
            np.random.seed(constants.random_seed())
            sample_indices = np.random.choice(a = np.arange(1000), size = n_sample, replace = False)
            ds = ds["train"]
            self.load_data(ds, sample_indices)
        elif iden == "Tune":
            ds = ds["train"]
            np.random.seed(constants.random_seed())

            filter_indexes = filtered_indices(ds["question"], ds["context"], self.tokenizer)
            # https://huggingface.co/docs/datasets/process
            ds = ds.filter(lambda example, idx: idx in filter_indexes, with_indices=True)
            print("Filtered Length of ds = {}".format(len(ds)))

            sample_indices, extra_contexts = self.ds_sample_indices(ds)
            self.load_data(ds, sample_indices, extra_contexts)
        elif iden == "Test":
            ds = ds["validation"]
            np.random.seed(constants.random_seed())

            filter_indexes = filtered_indices(ds["question"], ds["context"], self.tokenizer)
            # https://huggingface.co/docs/datasets/process
            ds = ds.filter(lambda example, idx: idx in filter_indexes, with_indices=True)
            print("Filtered Length of ds = {}".format(len(ds)))

            sample_indices, extra_contexts = self.ds_sample_indices(ds)
            self.load_data(ds, sample_indices, extra_contexts)
        elif iden == "TrainFull":
            ds = ds["train"]
            self.load_data(ds)
        elif iden == "TestFull":
            ds = ds["validation"]
            self.load_data(ds)
        else:
            raise Exception(f"Unknown Data Identifier {iden} in Squad Dataset")
        
    
    def ds_sample_indices(self, ds):
        # Sample 500 contexts and 5 queries per context. 
        n = constants.squad_sample_contexts_size()
        k = constants.squad_sample_questions_per_contexts()
        N = constants.squad_corpus_size()

        contexts = list(ds["context"])
        self.N = n * k 

        sample_indices, extra_contexts = sample_by_contexts(contexts, k, n, N)

        return sample_indices, extra_contexts


    def load_data(self, ds, sample_indices = None, extra_contexts = None):
        self.questions = list(ds["question"])
        self.contexts = list(ds["context"])
        self.answers = [lst[0] for lst in ds["answers"]["text"]]
        self.answer_starts = [lst[0] for lst in ds["answers"]["answer_start"]]
        self.answer_starts_relative = [self.answer_starts[index] / len(self.contexts[index]) for index in range(len(self.contexts))]
        self.N = len(self.questions)
        assert(self.N == len(self.contexts))
        assert(self.N == len(self.answers))

        if sample_indices is not None:
            self.questions = [self.questions[index] for index in sample_indices]
            self.answer_starts_relative = [self.answer_starts[index] / len(self.contexts[index]) for index in sample_indices]
            self.answer_starts = [self.answer_starts[index] for index in sample_indices]
            self.contexts = [self.contexts[index] for index in sample_indices]
            self.answers = [self.answers[index] for index in sample_indices]
            self.N = len(sample_indices)
        else:
            self.N = len(ds)

        sorted_order = find_length_argsort(self.contexts) 
        (questions, contexts, answers, answer_starts) = self.convert_to_sorted_order(self.questions,
                                                                                           self.contexts, self.answers,
                                                                                           self.answer_starts, sorted_order)
        self.questions = questions 
        self.contexts = contexts

        if not(extra_contexts is None):
            self.corpus = list_distinct(self.contexts) + extra_contexts
            
        self.answers = answers 
        self.answer_starts = answer_starts 
        self.qids = [i for i in range(len(self.questions))]
        self.question_types = ["Span" for i in range(len(self.questions))]


    def convert_to_sorted_order(self, questions, contexts, answers, answer_starts, arg_sort):
        questions = list(np.array(questions)[arg_sort])
        contexts = list(np.array(contexts)[arg_sort])
        answers = list(np.array(answers)[arg_sort])
        answer_starts = list(np.array(answer_starts)[arg_sort])

        return questions, contexts, answers, answer_starts

    def __len__(self):
        return self.N
    
    def __getitem__(self, index):
        return (self.questions[index], self.contexts[index], self.answers[index], self.qids[index])
    
    def get_questions(self):
        return self.questions
    
    def get_contexts(self):
        return self.contexts
    
    def get_answers(self):
        return self.answers
    
    def get_answer_starts(self):
        return self.answer_starts
    
    def get_relative_answer_starts(self):
        return self.answer_starts_relative
    
    def get_question_types(self):
        return self.question_types
    
    def get_qids(self):
        return self.qids
    
    def get_name(self):
        return f"Squad_{self.iden}"
    
    def get_corpus(self):
        return self.corpus


if __name__ == "__main__":
    data = Squad("Dummy")