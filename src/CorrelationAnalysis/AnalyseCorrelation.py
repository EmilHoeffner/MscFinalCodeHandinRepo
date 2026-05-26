from dataloader import DataLoader

import matplotlib.pyplot as plt

import os 
import sys

import pandas as pd

from scipy.stats import pearsonr
from scipy.stats import pointbiserialr
from scipy.stats import BootstrapMethod
import numpy as np

sys.path.append("..")

from Filehandling.FileHandler import FileHandler

from Utils import plotUtils


def correlation_pairs(variables, variable_names, dir_path):
    N_variables = len(variables)

    for i in range(N_variables - 1):
        current_variable_values = variables[i]
        current_variable_name = variable_names[i]
        for next_variable_vals, next_variable_name in zip(variables[i+1:], variable_names[i+1:]):
            correlation_plot(current_variable_values, current_variable_name, next_variable_vals
                                                  ,next_variable_name, dir_path)

            
def correlation_to_response(variables, variable_names, response, response_name, dir_path):
    for (variable_vals, variable_name) in zip(variables, variable_names):
        correlation_plot(variable_vals, variable_name, response, response_name, dir_path)

def correlation_plot(x, x_name, y, y_name, dir_path):
    cor = pearsonr(x, y, alternative = "two-sided")
    val = cor[0]
    p_val = cor[1]

    title = f"{x_name} vs {y_name}\ncor = {round(val,5)} at p = {round(p_val,5)}"
    x_label = x_name
    y_label = y_name
    save_path = dir_path + f"{x_name}_to_{y_name}.png"

    plotUtils.scatter_density_2d(x, y, title, x_label, y_label, save_path, log_color_bar = True)

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pointbiserialr.html
def binary_context_length_correlation(context_lengths, scores, y_name, dir_path, threshold = 512):
        scores_short = []
        scores_long = []
        for (context_length, score) in zip(context_lengths, scores):
            if context_length > threshold:
                scores_long.append(score)
            else:
                scores_short.append(score)


        X = [0 for _ in scores_short] + [1 for _ in scores_long]
        Y = [s for s in scores_short] + [s for s in scores_long]

        assert(len(X) == len(Y))

        cor = pointbiserialr(x = X, y = Y) 
        val = cor[0]
        p_val = cor[1]

        data_dict = {"Long Context({})".format(len(scores_long)) : scores_long, 
                     "Short Context({})".format(len(scores_short)) : scores_short}
        plt.title(f"cor = {round(val, 5)} with p value {round(p_val, 5)}")
        plt.boxplot(data_dict.values(), tick_labels=data_dict.keys())
        plt.savefig(dir_path + f"context_long_short_boxplot_{y_name}.png")
        plt.close()


def F1_score_0(f1_scores, edit_dist_scores, bert_scores, qids, data_set_iden, file_handler):
    arg_sort = np.argsort(qids)
    qids = list(np.array(qids)[arg_sort])
    f1_scores = list(np.array(f1_scores)[arg_sort])
    edit_dist_scores = list(np.array(edit_dist_scores)[arg_sort])
    bert_scores = list(np.array(bert_scores)[arg_sort])

    df_queries = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "queries.csv")
    df_queries = df_queries[df_queries["qid"].isin(qids)]
    df_queries_qids = list(df_queries["qid"])
    questions = list(df_queries["query"])

    df_answers = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "answers.csv")
    df_answers = df_answers[df_answers["qid"].isin(qids)]
    df_answers_qids = list(df_answers["qid"])
    true_answers = list(df_answers["answer"])

    df_results = file_handler.load_df(dir = "EvaluationFiles", dataset_name = data_set_iden, filename = "M_context.csv")
    df_results = df_results[df_results["qid"].isin(qids)]
    df_results_qids = list(df_results["qid"])
    predicted_answers = list(df_results["Predicted Answer"])

    assert(df_queries_qids == df_answers_qids)
    assert(df_answers_qids == df_results_qids)
    assert(df_results_qids == qids)

    questions_F1_0 = [q for i,q in enumerate(questions) if f1_scores[i] == 0]
    true_answers_F1_0 = [a for i,a in enumerate(true_answers) if f1_scores[i] == 0]
    predicted_answers_F1_0 = [a for i,a in enumerate(predicted_answers) if f1_scores[i] == 0]
    edit_dist_F1_0 = [e for i,e in enumerate(edit_dist_scores) if f1_scores[i] == 0]
    bert_F1_0 = [b for i,b in enumerate(bert_scores) if f1_scores[i] == 0]

    df = pd.DataFrame({"question" : questions_F1_0, "True Answer" : true_answers_F1_0, "Predicted Answer" : predicted_answers_F1_0,
                       "EditDist" : edit_dist_F1_0, "BERT" : bert_F1_0})
    
    return df

def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # https://www.nielit.gov.in/gorakhpur/sites/default/files/Gorakhpur/ALevel_1_Python_20May.pdf
    num_args = len(sys.argv)

    if num_args != 3:
        raise Exception("Program must be run with exactly two arguments")
    
    data_set_iden = sys.argv[1]
    y_name = sys.argv[2]

    dl = DataLoader(data_set_iden)
    file_handler = FileHandler()

    features, feature_names = dl.named_train_features()
    if y_name == "Add":
        y_train = [(f1 + bert) / 2.0 for f1,bert in zip(dl.load_train_feature("F1"), dl.load_train_feature("BERT"))]
        y_name = "0.5(F1+BERT)"
    else:
        y_train = dl.load_train_feature(y_name)

    
    '''
        Correlation between features (e.g. Context Length and Question Length)
    ''' 
    dir_path = file_handler.storage_path() + f"/CorrelationAnalysis/{data_set_iden}/Feature_to_Feature/"
    correlation_pairs(features, feature_names, dir_path)


    '''
        Correlation between different response variables (e.g. F1 and BERT)
    '''     
    dir_path = file_handler.storage_path() + f"/CorrelationAnalysis/{data_set_iden}/Response_to_Response/"
    response_names = ["Jaccard","Recall","Precision","F1","EditDist","EM","BERT"]
    response_values = [dl.load_train_feature(response_name) for response_name in response_names]
    correlation_pairs(response_values, response_names, dir_path)
    
    '''
        Correlation between every feature to the provided response variable.
        Also, makes box plot comparison with samples of short and long context.
    '''
    dir_path = file_handler.storage_path() + f"/CorrelationAnalysis/{data_set_iden}/Feature_to_Response/"
    
    # Level to Level Correlation:
    correlation_to_response(features, feature_names, y_train, y_name, dir_path)
    binary_context_length_correlation(dl.CL_train(), y_train,y_name, dir_path)
    
    # Some specific investigation with regards to the response variables
    f1_scores = dl.load_train_feature("F1")
    edit_dist_scores = dl.load_train_feature("EditDist")
    bert_scores = dl.load_train_feature("BERT")
    qids = dl.load_train_feature("qid")

    df = F1_score_0(f1_scores, edit_dist_scores, bert_scores, qids, data_set_iden, file_handler)
    file_handler.save_df(df = df, dir = "CorrelationAnalysis", dataset_name = data_set_iden, filename = "F1=0Adhoc.csv", header = True)


if __name__ == "__main__":
    main()


