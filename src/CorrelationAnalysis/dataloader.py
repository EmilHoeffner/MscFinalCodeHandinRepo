import sys 

import numpy as np
import pandas as pd

from transformers import AutoTokenizer

from tqdm import tqdm


sys.path.append("..")


from Filehandling.FileHandler import FileHandler
from TextAnalysis.TextAnalysis import lix, Automated_Readability ,Coleman_liau, Flesh_Kincaid, Gunning_Fox, Dale_Chall
from Evaluation.Evaluater import Evaluater

class DataLoader:

    def __init__(self, data_iden):
        file_handler = FileHandler()

        self.data_iden = data_iden


        if file_handler.file_exists(dir = "CorrelationAnalysis", dataset_name = data_iden,
                                    filename = "cache_train.csv") and file_handler.file_exists(dir = "CorrelationAnalysis", dataset_name = data_iden,filename = "cache_test.csv"):
            
            self.df_train = file_handler.load_df(dir = "CorrelationAnalysis", dataset_name = data_iden,
                                    filename = "cache_train.csv")
            self.df_test = file_handler.load_df(dir = "CorrelationAnalysis", dataset_name = data_iden,
                                    filename = "cache_test.csv")
        else:
            self.store_data(file_handler, data_iden)
            self.df_train = file_handler.load_df(dir = "CorrelationAnalysis", dataset_name = data_iden,
                                    filename = "cache_train.csv")
            self.df_test = file_handler.load_df(dir = "CorrelationAnalysis", dataset_name = data_iden,
                                    filename = "cache_test.csv")


    def store_data(self, file_handler, data_set_iden):
        df_query = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "queries.csv")
        df_answer = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "answers.csv")
        df_qrels = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "qrels.csv")
        df_docs = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "docs.csv")
        df_results = file_handler.load_df(dir = "EvaluationFiles", dataset_name = data_set_iden, filename = "M_context.csv")

        df_question = file_handler.load_df(dir = "EvaluationFiles", dataset_name = data_set_iden, filename = "M_question.csv")

        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.join.html
        # https://stackoverflow.com/questions/26645515/pandas-join-issue-columns-overlap-but-no-suffix-specified
        # https://stackoverflow.com/questions/19125091/pandas-merge-how-to-avoid-duplicating-columns
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html
        joined = df_query.merge(df_answer, on="qid", how = "left", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
        joined = joined.merge(df_qrels, on="qid", how = "left", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
        joined = joined.merge(df_results, on="qid", how = "left", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
        joined = joined.merge(df_docs, on="docno", how = "left", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

        # Split to train and testing, by first shuffling, and then choosing subsets.
        # https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
        joined = joined.sample(frac = 1, replace = False, random_state = 42)
        df_question = df_question.sample(frac = 1, replace = False, random_state = 42)

        N_train = np.int32(len(joined) * 0.8)

        train_set = joined[0:N_train]
        train_set_df_question = df_question[0:N_train]
        test_set = joined[N_train:]
        test_set_df_question = df_question[N_train:]

        questions_train = list(train_set["query"])
        contexts_train = list(train_set["text"])
        questions_test = list(test_set["query"])
        contexts_test = list(test_set["text"])

        evaluater = Evaluater()

        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl", device_map = "cuda")
        tokenizer.model_max_length = 4000

        # Qids:
        qids_train = list(train_set["qid"])

        # Readability metrics
        # Length
        QL_train = np.array([len(tokenizer.tokenize(question)) for question in questions_train])
        CL_train = np.array([len(tokenizer.tokenize(context)) for context in contexts_train])

        # Lix
        QX_train = np.array([lix(question) for question in questions_train])
        CX_train = np.array([lix(context) for context in contexts_train])

        # Automated Readability
        CAR_train = np.array([Automated_Readability(context) for context in contexts_train])
        # Coleman Liau
        CCL_train = np.array([Coleman_liau(context) for context in contexts_train])
        # Flesh Kincaid
        CFK_train = np.array([Flesh_Kincaid(context) for context in contexts_train])
        # Gunning Fox
        CGF_train = np.array([Gunning_Fox(context) for context in contexts_train])
        # Dale Chall 
        CDC_train = np.array([Dale_Chall(context) for context in contexts_train])

        Sim_train = np.array([evaluater.BERT_score(question, context) for (question, context) in zip(questions_train, contexts_train)])
        print("Computing Maximal BERT scores:")
        Sim_Max_train = np.array([self.maximal_bert(question, context, evaluater) for (question, context) in tqdm(zip(questions_train, contexts_train))])
        AS_train = np.array(list(train_set["answer_start"]))
        AS_relative_train = np.array(list(train_set["answer_start_relative"]))
        PA_train = np.array(list(train_set["Perp_ans"]))
        QT_train = np.array([self.question_type_to_numeric(q_type) for q_type in list(train_set["type"])])

        y_question_train_F1 = np.array(list(train_set_df_question["F1"]))
        y_question_train_BERT = np.array(list(train_set_df_question["BERT"]))
        y_question_train_EditDist = np.array(list(train_set_df_question["EditDist"]))


        qids_test = list(test_set["qid"])
        QL_test = np.array([len(tokenizer.tokenize(question)) for question in questions_test])
        CL_test = np.array([len(tokenizer.tokenize(context)) for context in contexts_test])
        CDC_test = np.array([Dale_Chall(context) for context in contexts_test])
        PA_test = np.array(list(test_set["Perp_ans"]))
        Sim_Max_test = np.array([self.maximal_bert(question, context, evaluater) for (question, context) in tqdm(zip(questions_test, contexts_test))])

        y_question_test_F1 = np.array(list(test_set_df_question["F1"]))
        y_question_test_BERT = np.array(list(test_set_df_question["BERT"]))
        y_question_test_EditDist = np.array(list(test_set_df_question["EditDist"]))

        response_names = ["Jaccard","Recall","Precision","F1","EditDist","EM","BERT"]
        
        df_train = pd.DataFrame({"qid": qids_train, "question" : questions_train, "context" : contexts_train,
                                 "QL" : QL_train, "CL" : CL_train, "QX" : QX_train, "CX" : CX_train,
                                 "CAR" : CAR_train, "CCL" : CCL_train, "CFK" : CFK_train,
                                 "CGF" : CGF_train, "CDC" : CDC_train, "Sim" : Sim_train,
                                 "Sim_Max" : Sim_Max_train, "AS" : AS_train, 
                                 "AS_relative" : AS_relative_train, "PA" : PA_train,
                                 "QT" : QT_train,
                                 "y_question_F1" : y_question_train_F1, 
                                 "y_question_BERT" : y_question_train_BERT,
                                 "y_question_EditDist" : y_question_train_EditDist}) 
        
        # https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html
        df_response_train = train_set[["qid"] + response_names]
        
        df_train = df_train.merge(df_response_train, on="qid", how = "left", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

        
        df_test = pd.DataFrame({"qid": qids_test, "question" : questions_test, "context" : contexts_test,
                                "QL" : QL_test, "CL" : CL_test, "CDC" : CDC_test,
                                "PA" : PA_test, "Sim_Max" : Sim_Max_test,
                                "y_question_F1" : y_question_test_F1,
                                "y_question_test_BERT" : y_question_test_BERT,
                                "y_question_test_EditDist" : y_question_test_EditDist})
        
        df_response_test = test_set[["qid"] + response_names]

        df_test = df_test.merge(df_response_test, on="qid", how = "left", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

        
        file_handler.save_df(df = df_train, dir = "CorrelationAnalysis", 
                             dataset_name = data_set_iden, filename = "cache_train.csv", header = True)
        

        file_handler.save_df(df = df_test, dir = "CorrelationAnalysis", 
                             dataset_name = data_set_iden, filename = "cache_test.csv", header = True)
        
    def named_train_features(self):
        feature_names = ["Question Length", "Context Length", "Question Lix", "Context Lix",
                     "Automated Read", "Coleman Liau", "Flesh Kincaid",
                     "Gunning Fox", "Dale Chall",
                     "Sim", "Sim_Max", "Answer_Start", "Answer_Start_Relative", 
                     "Perplexity", "y_question_F1", "y_question_BERT",
                      "y_question_EditDist", "question_type"
                    ]
        
        features = [self.QL_train(), self.CL_train(), self.QX_train(), 
                    self.CX_train(), self.CAR_train(), self.CCL_train(), 
                    self.CFK_train(), self.CGF_train(), self.CDC_train(), 
                    self.Sim_train(), self.Sim_Max_train(), self.AS_train(), 
                    self.AS_relative_train(), self.PA_train(), 
                    self.y_question_F1_train(), self.y_question_BERT_train(),
                    self.y_question_EditDist_train(), self.QT_train()]

        assert(len(features) == len(feature_names))

        return features, feature_names
        
    
    def load_test_feature(self, iden):
        return list(self.df_test[iden])
    
    def load_train_feature(self, iden):
        return list(self.df_train[iden])
    

    def QL_test(self):
        return list(self.df_test["QL"])
    
    def CL_test(self):
        return list(self.df_test["CL"])
    
    def CDC_test(self):
        return list(self.df_test["CDC"])
    
    def PA_test(self):
        return list(self.df_test["PA"])
    
    def y_question_F1_test(self):
        return list(self.df_test["y_question_F1"])
    
    def y_question_BERT_test(self):
        return list(self.df_test["y_question_BERT"])
    
    def y_question_EditDist_test(self):
        return list(self.df_test["y_question_EditDist"])

    def QL_train(self):
        return list(self.df_train["QL"])
    
    def CL_train(self):
        return list(self.df_train["CL"])
    
    def QX_train(self):
        return list(self.df_train["QX"])

    def CX_train(self):
        return list(self.df_train["CX"])
    
    def CAR_train(self):
        return list(self.df_train["CAR"])
    
    def CCL_train(self):
        return list(self.df_train["CCL"])
    
    def CFK_train(self):
        return list(self.df_train["CFK"])
    
    def CGF_train(self):
        return list(self.df_train["CGF"])

    def CDC_train(self):
        return list(self.df_train["CDC"])
    
    def Sim_train(self):
        return list(self.df_train["Sim"])
    
    def Sim_Max_train(self):
        return list(self.df_train["Sim_Max"])
    
    def AS_train(self):
        return list(self.df_train["AS"])
    
    def AS_relative_train(self):
        return list(self.df_train["AS_relative"])
    
    def PA_train(self):
        return list(self.df_train["PA"])
    
    def QT_train(self):
        return list(self.df_train["QT"])
    
    def y_question_F1_train(self):
        return list(self.df_train["y_question_F1"])
    
    def y_question_BERT_train(self):
        return list(self.df_train["y_question_BERT"])
    
    def y_question_EditDist_train(self):
        return list(self.df_train["y_question_EditDist"])
    

    def maximal_bert(self, question, context, evaluater):
        sentences = context.split(".")
        num_sences = len(sentences)

        # To exclude empty string, due to sentence ending on punctuation
        sentences = sentences[0:num_sences-1]

        max_BERT_score = 0

        for sentence in sentences:
            BERT_score = evaluater.BERT_score(question, sentence)

            if BERT_score > max_BERT_score:
                max_BERT_score = BERT_score 

        return max_BERT_score

    def question_type_to_numeric(self, question_type):
        question_type = question_type.lower()
        if question_type == "span":
            return -1
        elif question_type == "comparison":
            return 0
        elif question_type == "bridge_comparison":
            return 1
        elif question_type == "compositional":
            return 2
        elif question_type == "inference":
            return 3
        else:
            raise Exception("Got Unknown Question Type: {}".format(question_type))


