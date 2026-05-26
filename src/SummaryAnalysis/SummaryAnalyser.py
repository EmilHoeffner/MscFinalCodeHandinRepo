
'''
    A class for computing summary statistics and confidence intervals for models in the end-to-end tuning and main experiments.
'''

import sys 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("..")

from Filehandling.FileHandler import FileHandler

class SummaryAnalyser:

    def __init__(self, data_iden):
        self.data_iden = data_iden  
        self.file_handler = FileHandler()

    def pruned_summary_analysis(self):
        df_context = self.file_handler.load_df(dir = "EvaluationFiles", dataset_name = self.data_iden, filename = "M_context.csv")
        df_pruned = self.file_handler.load_df(dir = "EvaluationFiles", dataset_name = self.data_iden, filename = "M_context_pruned.csv")
        bad_qids = list(self.file_handler.load_df(dir = "DataFiles", dataset_name = self.data_iden, filename = "BadQids.csv")["qid"])

        qids = list(df_context["qid"])
        df_context_answers = list(df_context["Predicted Answer"])
        df_pruned_answers = list(df_pruned["Predicted Answer"])

        
        '''
        for qid, orig_ans, pruned_ans in zip(qids, df_context_answers, df_pruned_answers):
            orig_ans = str(orig_ans)
            pruned_ans = str(pruned_ans)

            if orig_ans == "IDK" or orig_ans == "idk" or pruned_ans == "IDK" or pruned_ans == "idk":
                bad_qids.append(qid)
        '''
        
        
        print(f"Number of removed qids = {len(bad_qids)}")

        # https://stackoverflow.com/questions/19960077/how-to-filter-pandas-dataframe-using-in-and-not-in-like-in-sql
        df_context = df_context[~df_context.qid.isin(bad_qids)]
        df_pruned = df_pruned[~df_pruned.qid.isin(bad_qids)]

        print(len(df_context))
        print(len(df_pruned))

        # Only take columns after the first four columns (these contain the metrics)
        metric_names = df_context.columns.values.tolist()[3:]
        
        column_names = ["M_Ceil", "M_Pruned"]
        # https://stackoverflow.com/questions/52780277/how-do-i-name-columns-rows-in-pandas-dataframe-for-clarity
        row_names = []
        row_values = []

        for metric_name in metric_names:
            vals_context = np.array(list(df_context[metric_name]))
            vals_pruned = np.array(list(df_pruned[metric_name]))

            means = [np.mean(vals_context), np.mean(vals_pruned)]
            stds =  [np.std(vals_context), np.std(vals_pruned)]
                
            row_names.append(f"{metric_name}_mean")
            row_names.append(f"{metric_name}_std")

            row_values.append(means)
            row_values.append(stds)

        df = pd.DataFrame(row_values, row_names, column_names)
        self.file_handler.save_df(df = df, dir = "SummaryAnalysis", dataset_name = self.data_iden, filename = "SummaryAnalysis_Pruned.csv", header = True, index = True)


        
    def summary_model_analysis(self):
        df_question = self.file_handler.load_df(dir = "EvaluationFiles", dataset_name = self.data_iden, filename = "M_question.csv")
        df_RAG = self.file_handler.load_df(dir = "EvaluationFiles", dataset_name = self.data_iden, filename = "M_RAG.csv")
        df_context = self.file_handler.load_df(dir = "EvaluationFiles", dataset_name = self.data_iden, filename = "M_context.csv")
        # https://stackoverflow.com/questions/19482970/get-a-list-from-pandas-dataframe-column-headers

        # Only take columns after the first four columns (these contain the metrics)
        metric_names = df_question.columns.values.tolist()[3:]
        
        column_names = ["M_Question", "M_RAG", "M_Context"]
        # https://stackoverflow.com/questions/52780277/how-do-i-name-columns-rows-in-pandas-dataframe-for-clarity
        row_names = []
        row_values = []

        for metric_name in metric_names:
            vals_question = np.array(list(df_question[metric_name]))
            vals_RAG = np.array(list(df_RAG[metric_name]))
            vals_context = np.array(list(df_context[metric_name]))

            means = [np.mean(vals_question), np.mean(vals_RAG), np.mean(vals_context)]
            stds =  [np.std(vals_question), np.std(vals_RAG), np.std(vals_context)]
                
            row_names.append(f"{metric_name}_mean")
            row_names.append(f"{metric_name}_std")

            row_values.append(means)
            row_values.append(stds)

        df = pd.DataFrame(row_values, row_names, column_names)
        self.file_handler.save_df(df = df, dir = "SummaryAnalysis", dataset_name = self.data_iden, filename = "SummaryAnalysis.csv", header = True, index = True)

    def bootstrap_analysis(self):
        df_question = self.file_handler.load_df(dir = "EvaluationFiles", dataset_name = self.data_iden, filename = "M_question.csv")
        df_RAG = self.file_handler.load_df(dir = "EvaluationFiles", dataset_name = self.data_iden, filename = "M_RAG.csv")
        df_ceil = self.file_handler.load_df(dir = "EvaluationFiles", dataset_name = self.data_iden, filename = "M_context.csv")

        metric_names = df_question.columns.values.tolist()[3:]

        row_values = []

        lower_percentile_func = lambda x: np.sort(x)[np.int32(0.25 * len(x))]
        median_func = lambda x: np.median(np.sort(x))
        upper_percentile_func = lambda x: np.sort(x)[np.int32(0.75 * len(x))]

        bootstrap_functions = [np.mean, np.std ,np.amin, lower_percentile_func, median_func, upper_percentile_func, np.amax]
        column_names = ["Mean", "std", "Min", "0.25q", "median", "0.75q", "max"]

        row_values = []
        row_names = metric_names

        dfs = [df_question, df_RAG, df_ceil]
        df_names = ["Question", "RAG", "Ceil"]

        for (df, df_name) in zip(dfs, df_names):
            row_values = []
            for metric_name in metric_names:
                lst = []
                vals = np.array(list(df[metric_name]))
                for bootstrap_func in bootstrap_functions:
                    CI = self.bootstrap(vals, bootstrap_func)
                    CI_string = f"[{CI[0],CI[1]}]"
                    lst.append(CI_string)
                row_values.append(lst)

            df = pd.DataFrame(row_values, row_names, column_names)
            self.file_handler.save_df(df = df, dir = "SummaryAnalysis", dataset_name = self.data_iden, filename = f"Bootstrap_{df_name}.csv", header = True, index = True)

    '''
        x: An np array of samples
        bootstrap_stats_function: A function, that takes an np array and returns the relevant statistic
    '''
    def bootstrap(self, x, bootstrap_stats_function, alpha = 0.05):
        repetions = 10000
        estimations = []
        N = len(x)

        for i in range(repetions):
            samples = np.random.choice(a = x, size = N, replace = True)
            estimations.append(bootstrap_stats_function(samples))
        
        estimations.sort()

        #plt.hist(x = estimations, bins = 100)
        #plt.show()

        lower_sig = alpha / 2.0
        upper_sig = 1.0 - lower_sig

        lower = estimations[np.int32(repetions * lower_sig)]
        upper = estimations[np.int32(repetions * upper_sig)]

        return [float(lower), float(upper)]
            





        



