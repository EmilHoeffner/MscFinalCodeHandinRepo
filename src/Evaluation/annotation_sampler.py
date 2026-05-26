import pandas as pd
import sys 
import numpy as np

sys.path.append("..")

from Filehandling.FileHandler import FileHandler
from Constants import constants


def main():
    # https://www.nielit.gov.in/gorakhpur/sites/default/files/Gorakhpur/ALevel_1_Python_20May.pdf
    num_args = len(sys.argv)

    if num_args != 2:
        raise Exception("Program must be run with exactly one argument, denoting the data split")
    
    data_set_iden = sys.argv[1]

    file_handler = FileHandler()

    df_queries = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "queries.csv")
    df_queries = df_queries[["qid", "query"]]

    df_qrels = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "qrels.csv")
    df_docs = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "docs.csv")

    df_documentData = df_docs.merge(df_qrels[["qid", "docno"]], on="docno", how = "left", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

    df_question = file_handler.load_df(dir = "EvaluationFiles", dataset_name = data_set_iden, filename = "M_question.csv")
    df_RAG = file_handler.load_df(dir = "EvaluationFiles", dataset_name = data_set_iden, filename = "M_RAG.csv")
    df_context = file_handler.load_df(dir = "EvaluationFiles", dataset_name = data_set_iden, filename = "M_context.csv")

    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html
    df_question = df_question.rename(columns = {"Predicted Answer" : "M_question"})
    df_RAG = df_RAG.rename(columns = {"Predicted Answer" : "M_RAG"})
    df_context = df_context.rename(columns = {"Predicted Answer" : "M_ceil"})

    n_sample = 50
    samples = df_queries.sample(n = n_sample, replace = False,
                                random_state = constants.random_seed()).reset_index(drop=True)
    
    # Modified from dataloader.py in CorrelationAnalysis
    # https://stackoverflow.com/questions/17978133/python-pandas-merge-only-certain-columns
    joined = samples.merge(df_question[["qid", "True Answer", "M_question"]], on="qid", how = "left", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
    joined = joined.merge(df_RAG[["qid", "M_RAG"]], on="qid", how = "left", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
    joined = joined.merge(df_context[["qid", "M_ceil"]], on="qid", how = "left", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

    # https://www.geeksforgeeks.org/python/how-to-add-empty-column-to-dataframe-in-pandas/
    joined["M_Question_Rank"] = "NA"
    joined["M_Question_Halu"] = "NA"
    joined["M_RAG_Rank"] = "NA"
    joined["M_RAG_Halu"] = "NA"
    joined["M_Ceil_Rank"] = "NA"
    joined["M_Ceil_Halu"] = "NA"

    # Add the Relevant context
    joined = joined.merge(df_documentData[["qid", "text"]], on="qid", how = "left", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

    file_handler.save_df(df = joined, dir = "EvaluationFiles", dataset_name = data_set_iden,
                          filename = "annotation.csv", header = True)













if __name__ == "__main__":
    main()