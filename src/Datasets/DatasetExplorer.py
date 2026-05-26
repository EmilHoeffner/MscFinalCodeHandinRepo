from transformers import AutoTokenizer

import matplotlib.pyplot as plt

import sys

sys.path.append("..")

from Filehandling.FileHandler import FileHandler
from TextAnalysis.TextAnalysis import lix
from TextAnalysis.TextAnalysis import word_cloud
from Constants import constants

class DatasetExplorer:

    def __init__(self, dataset_iden, dataset):
        self.dataset = dataset 
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
        self.tokenizer.model_max_length = constants.tokenizer_max_length()
        self.data_name = dataset.get_name()

        self.save_path = FileHandler().storage_path() + f"/DatasetStatistics/{dataset_iden}/"
    
    def store_context_token_length_distribution(self):
        if self.data_name == "Squad_TrainFull" or self.data_name == "Squad_TestFull":
            contexts = self.dataset.get_contexts()
        else:
            contexts = self.dataset.get_corpus()
        token_lengths = [self.num_tokens(context) for context in contexts]
        self.store_boxplot(values = token_lengths, y_label = "Length in tokens", 
                           title = f"{self.dataset.get_name()}: Context Length Distribution", save_name = "ContextLength")
        
    def store_answer_token_length_distribution(self):
        answers = self.dataset.get_answers()
        token_lengths = [self.num_tokens(answer) for answer in answers]
        #self.store_boxplot(values = token_lengths, y_label = "Length in tokens", 
                           #title = f"{self.dataset.get_name()}: Answer Length Distribution", 
                           #save_name = "AnswerLength")
        
        possible_lengths = set(token_lengths)
        dict = {}

        for l in possible_lengths:
            dict[l] = 0
        

        for tl in token_lengths:
            dict[tl] += 1

        x = list(dict.keys())
        heights = list(dict.values())

        save_path = self.save_path + f"AnswerLength.png"
        title = self.dataset.get_name() + ": Answer Length Distribution"

        plt.title(title, fontsize = 16)
        plt.bar(x, heights)
        plt.ylabel("Frequency", fontsize = 18)
        plt.xlabel("Length in tokens", fontsize = 18)
        plt.tick_params("x", labelsize = 18)
        plt.tick_params("y", labelsize = 18)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


    def store_context_lix_distribution(self):
        if self.data_name == "Squad_TrainFull" or self.data_name == "Squad_TestFull":
            contexts = self.dataset.get_contexts()
        else:
            contexts = self.dataset.get_corpus()
        difficulties = [lix(context) for context in contexts]
        self.store_boxplot(values = difficulties, y_label = "Lix", 
                           title = f"{self.dataset.get_name()}: Context Lix Distribution", save_name = "ContextLix")

    def store_context_word_cloud(self):
        if self.data_name == "Squad_TrainFull" or self.data_name == "Squad_TestFull":
            contexts = self.dataset.get_contexts()
        else:
            contexts = self.dataset.get_corpus()
        word_cloud(contexts, save_path = self.save_path + "Context_Word_Cloud.png", title = f"{self.dataset.get_name()}: Question Word Cloud", max_words = 100)

    def store_questions_per_context(self):
        contexts = self.dataset.get_contexts()
        if self.data_name == "Squad_TrainFull" or self.data_name == "Squad_TestFull":
            unique_contexts = self.dataset.get_contexts()
        else:
            unique_contexts = self.dataset.get_corpus()

        dict = {}

        for context in unique_contexts:
            dict[context] = 0

        for context in contexts:
            val = dict[context]
            dict[context] = val + 1

        self.store_barchart(dict, y_label = "Frequency", x_label = "Triplets per Context", 
                            title = f"{self.dataset.get_name()}: Triplets per Context",
                            save_name = "QuestionPerContext")

    def store_question_token_length_distribution(self):
        questions = self.dataset.get_questions()
        token_lengths = [self.num_tokens(question) for question in questions]
        self.store_boxplot(values = token_lengths, y_label = "Length in tokens", 
                           title = f"{self.dataset.get_name()}: Question Length Distribution",
                           save_name = "QuestionLength")

    def store_question_lix_distribution(self):
        questions = self.dataset.get_questions()
        difficulties = [lix(question) for question in questions]
        self.store_boxplot(values = difficulties, y_label = "Lix", title = f"{self.dataset.get_name()}: Question Lix Distribution",
                           save_name = "QuestionLix")

    def store_question_word_cloud(self):
        questions = self.dataset.get_questions()
        word_cloud(questions, save_path = self.save_path + "Question_Word_Cloud.png", title = f"{self.dataset.get_name()}: Question Word Cloud", max_words = 100)

    # https://www.geeksforgeeks.org/data-visualization/box-plot-in-python-using-matplotlib/
    def store_boxplot(self, values, y_label, title, save_name):
        save_path = self.save_path + f"{save_name}.png"
        plt.title(title, fontsize = 18)
        plt.boxplot(values)
        plt.ylabel(y_label, fontsize = 18)
        plt.tick_params("x", labelsize = 18)
        plt.tick_params("y", labelsize = 18)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def store_barchart(self, dict, y_label, x_label, title, save_name):
        save_path = self.save_path + f"{save_name}.png"
        keys = list(dict.keys())
        buckets = list(set([dict[key] for key in keys]))

        buckets.sort()

        frequency_dict = {}

        for buck in buckets:
            frequency_dict[buck] = 0

        for key in keys:
            freq_keys = list(frequency_dict.keys())
            buck = dict[key]
            assert(buck in freq_keys)

            frequency_dict[buck] += 1

        x = list(frequency_dict.keys())
        heights = list(frequency_dict.values())

        plt.title(title, fontsize = 18)
        plt.bar(x, heights)
        plt.ylabel(y_label, fontsize = 18)
        plt.xlabel(x_label, fontsize = 18)
        plt.tick_params("x", labelsize = 18)
        plt.tick_params("y", labelsize = 18)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    '''
        sentence: A string

        Returns: Number of tokens in the sentence
    '''
    def num_tokens(self, sentence):
        return len(self.tokenizer.tokenize(sentence))


