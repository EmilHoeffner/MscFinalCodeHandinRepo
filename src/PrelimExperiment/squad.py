# https://huggingface.co/datasets/rajpurkar/squad?library=datasets

# https://www.geeksforgeeks.org/python/get-list-of-column-headers-from-a-pandas-dataframe/

# https://stackoverflow.com/questions/71102654/huggingface-datasets-convert-a-dataset-to-pandas-and-then-convert-it-back

from datasets import load_dataset
import numpy as np

class SQUAD:
    '''
        Convert to dataset consisiting of triplets:
        (Question, Context, Answer).
    '''
    def __init__(self):
        ds = load_dataset("rajpurkar/squad")
        ds = ds["train"]
        print("Columns = {}".format(ds.to_pandas().columns))
        self.questions = list(ds["question"])
        self.contexts = list(ds["context"])
        self.answers = [lst[0] for lst in ds["answers"]["text"]]
        self.answer_starts = [lst[0] for lst in ds["answers"]["answer_start"]]
        print(np.unique(np.array(self.answer_starts)))
        self.N = len(self.questions)
        print("N_train = {}".format(self.N))
        assert(self.N == len(self.contexts))
        assert(self.N == len(self.answers))

    def get_contexts_length_distribution(self):
        contexts = self.get_contexts()
        lengths = [len(context) for context in contexts]
        lengths.sort()
        print(lengths)

    def __len__(self):
        return self.N
    
    def __getitem__(self, index):
        return (self.questions[index], self.contexts[index], self.answers[index])
    
    def get_questions(self):
        return self.questions
    
    def get_contexts(self):
        return self.contexts
    
    def get_answers(self):
        return self.answers
    
    def train_val_cross_section(self):
        ds = load_dataset("rajpurkar/squad")
        ds = ds["validation"]

        questions_val = list(ds["question"])
        contexts_val = list(ds["context"])
        answers_val = [lst[0] for lst in ds["answers"]["text"]]

        questions_train = self.get_questions()
        contexts_train = self.get_contexts()
        answers_train = self.get_answers()

        self.N_val = len(questions_val)
        print("N_val = {}".format(self.N_val))

        question_overlap = [question for question in questions_val if question in questions_train]
        context_overlap = [context for context in contexts_val if context in contexts_train]

        print("Question Overlap = {}".format(len(question_overlap)))
        print("Context Overlap = {}".format(len(context_overlap)))





        
