
import sys

from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append("..")

from Datasets.datautils import find_length_argsort 
from Datasets.datautils import filtered_indices
from Constants import constants

# https://huggingface.co/datasets/framolfese/2WikiMultihopQA

class WikiMultiHop:

    def __init__(self, iden):
        ds = load_dataset("framolfese/2WikiMultihopQA") 
        self.iden = iden
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
        self.tokenizer.model_max_length = constants.tokenizer_max_length()


        if iden == "Dummy":
            # Using same samples as the preliminary experiment:
            n_sample = 200
            np.random.seed(33)
            self.N = 1000
            sample_indices = np.random.choice(a = np.arange(self.N), size = n_sample, replace = False)
            ds = ds["validation"]
            self.load_data(ds, sample_indices)
        elif iden == "Tune":
            n_sample = constants.dataset_sample_size()
            self.N = n_sample
            np.random.seed(constants.random_seed())

            ds = ds["validation"]
            sentences = list(ds["context"]["sentences"])
            contexts = [self.join_wikihopsentences_to_context(s) for s in sentences]
            questions = list(ds["question"])

            filter_indexes = filtered_indices(questions, contexts, self.tokenizer)
            # https://huggingface.co/docs/datasets/process
            ds = ds.filter(lambda example, idx: idx in filter_indexes, with_indices=True)

            sample_indices = np.random.choice(a = len(ds), size = n_sample, replace = False)

            #print("Tuning Sample Indices: {}".format(sample_indices))
            self.load_data(ds, sample_indices)
        elif iden == "Test":
            n_sample = constants.dataset_sample_size()
            self.N = n_sample
            np.random.seed(constants.random_seed())

            ds = ds["validation"]

            sentences = list(ds["context"]["sentences"])
            contexts = [self.join_wikihopsentences_to_context(s) for s in sentences]
            questions = list(ds["question"])

            filter_indexes = filtered_indices(questions, contexts, self.tokenizer)
            # https://huggingface.co/docs/datasets/process
            ds = ds.filter(lambda example, idx: idx in filter_indexes, with_indices=True)

            sample_indices_tune = np.random.choice(a = len(ds), size = n_sample, replace = False)

            #print("Tuning Sample Indices: {}".format(sample_indices_tune))

            sample_space_test = [i for i in range(len(ds)) if i not in sample_indices_tune]

            #print("Tuning Indices Test = {}".format(sample_space_test))
            sample_indices_test = np.random.choice(a = sample_space_test, size = n_sample, replace = False)

            self.load_data(ds, sample_indices_test)
        elif iden == "ValidationFull":
            ds = ds["validation"]
            self.load_data(ds)
        else:
            raise Exception(f"Unknown Data Identifier {iden} in WikiMultiHop Dataset")
    
    def join_wikihopsentences_to_context(self, s):
        new_contexts = []
        for context in s:
            joined_context = ""
            for sentence in context:
                joined_context += sentence + " "
            new_contexts.append(joined_context)
        
        final = ""

        for context in new_contexts:
            final += context 
        
        return final

    def load_data(self, ds, sample_indices = None):
        self.questions = list(ds["question"])
        self.answers = list(ds["answer"])
        self.question_types = list(ds["type"])
        
        contexts = ds["context"]
        self.contexts_sentences = list(contexts["sentences"])
        self.contexts_titles = list(contexts["title"])

        supporing_facts = ds["supporting_facts"]

        supporting_titles_total = list([s["title"] for s in supporing_facts])
        supporting_sentences_total = list([s["sent_id"] for s in supporing_facts])

        self.supporting_contexts = []
        self.distractive_contexts = []
        self.supportive_sentences = []

        for (context_passages, context_titles, supporting_titles, supporting_sentences) in zip(self.contexts_sentences, self.contexts_titles, supporting_titles_total, supporting_sentences_total):
            support = []
            support_sentences = []
            distractive = []
            assert(len(context_titles) == len(context_passages))
            counter_relevant_titles = 0
            for (title, passage) in zip(context_titles, context_passages):
                if title in supporting_titles:
                    support.append(passage)
                    support_sentences.append([passage[i] for i in range(len(passage)) if i == int(supporting_sentences[counter_relevant_titles])])
                    counter_relevant_titles += 1
                else:
                    distractive.append(passage)
            
            self.supporting_contexts.append(support)
            self.supportive_sentences.append(support_sentences)
            self.distractive_contexts.append(distractive)


        if sample_indices is not None:
            self.questions = [self.questions[index] for index in sample_indices]
            self.contexts_sentences = [self.contexts_sentences[index] for index in sample_indices]
            self.answers = [self.answers[index] for index in sample_indices]
            self.question_types = [self.question_types[index] for index in sample_indices]
            self.supporting_contexts = [self.supporting_contexts[index] for index in sample_indices]
            self.distractive_contexts = [self.distractive_contexts[index] for index in sample_indices]
            self.supportive_sentences = [self.supportive_sentences[index] for index in sample_indices]
            self.N = len(sample_indices)
        else:
            self.N = len(ds)

        # Join the contexts
        self.contexts = [self.join_wikihopsentences_to_context(context) for context in tqdm(self.contexts_sentences)]

        sorted_order = find_length_argsort(self.contexts) 
        (questions, contexts, supporting_contexts, supporting_sentences, distractive_contexts, answers, question_types) = self.convert_to_sorted_order(self.questions, 
                                                                                                                                 self.contexts, self.supporting_contexts, 
                                                                                                                                 self.supportive_sentences,
                                                                                                                                  self.distractive_contexts,
                                                                                                                                    self.answers, self.question_types, sorted_order)
        self.questions = questions 
        self.contexts = contexts
        self.supporting_contexts = supporting_contexts
        self.supportive_sentences = supporting_sentences
        self.distractive_contexts = distractive_contexts
        self.answers = answers 
        self.question_types = question_types

        self.answer_starts = [-1 for i in range(self.N)]
        self.answer_starts_relative = [-1 for i in range(self.N)]
        self.qids = [i for i in range(self.N)] 

    def convert_to_sorted_order(self, questions, contexts, supporting_contexts, supporting_sentences, distractive_contexts, answers, question_types, arg_sort):
        questions = list(np.array(questions)[arg_sort])
        contexts = list(np.array(contexts)[arg_sort])
        supporting = [supporting_contexts[sorted_idx] for sorted_idx in arg_sort]
        supporting_sentences = [supporting_sentences[sorted_idx] for sorted_idx in arg_sort]
        distractive = [distractive_contexts[sorted_idx] for sorted_idx in arg_sort]
        answers = list(np.array(answers)[arg_sort])
        question_types = list(np.array(question_types)[arg_sort])

        return questions, contexts, supporting, supporting_sentences, distractive, answers, question_types
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, index):
        return (self.questions[index], self.contexts[index], self.answers[index], self.qids[index])
    
    def get_questions(self):
        return self.questions
    
    def get_contexts(self):
        return self.contexts
    
    def get_supporting_contexts(self):
        return self.supporting_contexts
    
    def get_supporting_sentences(self):
        return self.supportive_sentences
    
    def get_distractive_contexts(self):
        return self.distractive_contexts
    
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
        return "WikiMultiHop_{}".format(self.iden)
    
    def get_corpus(self):
        return self.contexts



def main():
    ds = WikiMultiHop("Test")

    questions = ds.get_questions()
    contexts = ds.get_contexts()
    support = ds.get_supporting_contexts()
    support_sentences = ds.get_supporting_sentences() 
    distractive = ds.get_distractive_contexts()

    i = 0

    q = questions[i]
    s = support[i]
    ss = support_sentences[i]
    d = distractive[i]

    print("Question = {}".format(q))

    print("\nSupport:\n{}\n".format(s))

    print("\nSupport_Sentences:\n{}\n".format(ss))

    print("\nDistractive:\n{}".format(d))

if __name__ == "__main__":
    main()