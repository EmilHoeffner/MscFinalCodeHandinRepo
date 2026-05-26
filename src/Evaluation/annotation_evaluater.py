import pandas as pd
import sys 
import numpy as np
import random

sys.path.append("..")

from Filehandling.FileHandler import FileHandler
from Constants import constants

from SummaryAnalysis.SummaryAnalyser import SummaryAnalyser


def main():
    # https://www.nielit.gov.in/gorakhpur/sites/default/files/Gorakhpur/ALevel_1_Python_20May.pdf
    num_args = len(sys.argv)

    if num_args != 2:
        raise Exception("Program must be run with exactly one argument, denoting the data split")
    
    data_set_iden = sys.argv[1]

    file_handler = FileHandler()

    df = file_handler.load_df(dir = "EvaluationFiles", dataset_name = data_set_iden, filename = "annotation.csv")

    M_question_ranks = list(df["M_Question_Rank"])

    M_RAG_ranks = list(df["M_RAG_Rank"])
    M_RAG_halus = list(df["M_RAG_Halu"])

    M_ceil_ranks = list(df["M_Ceil_Rank"])
    M_ceil_halus = list(df["M_Ceil_Halu"])

    present_rank_distribution(M_question_ranks, "M_question")

    present_rank_distribution(M_RAG_ranks, "M_RAG")
    present_hallucination_distribution(M_RAG_halus, "M_RAG")

    present_rank_distribution(M_ceil_ranks, "M_ceil")
    present_hallucination_distribution(M_ceil_halus, "M_ceil")



def present_rank_distribution(lst, model_identifier):
    n_perfect = 0
    n_mid = 0
    n_bad = 0
    n_IDK = 0

    summary_analyser = SummaryAnalyser("")

    
    for val in lst:
        if str(val) == "IDK":
            n_IDK += 1
        elif float(val) == 1.0:
            n_perfect += 1
        elif float(val) == 0.5:
            n_mid += 1
        elif float(val) == 0.0:
            n_bad += 1
        else:
            raise Exception(f"Invalid Value {val}")
        
    avg_excluding_idk = (n_perfect * 1.0 + n_mid * 0.5 + n_bad * 0.0) / (n_perfect + n_mid + n_bad)
    avg_including_idk = (n_perfect * 1.0 + n_mid * 0.5 + n_bad * 0.0) / len(lst)


    scores_incl_idk = [1 for i in range(0,n_perfect)] + [0.5 for i in range(0,n_mid)] + [0.0 for i in range(0,n_bad + n_IDK)] 
    scores_excl_idk = [1 for i in range(0,n_perfect)] + [0.5 for i in range(0,n_mid)] + [0.0 for i in range(0,n_bad)] 

    # https://www.w3schools.com/python/ref_random_shuffle.asp
    random.shuffle(scores_incl_idk)
    random.shuffle(scores_excl_idk)

    CI_incl = summary_analyser.bootstrap(scores_incl_idk, bootstrap_stats_function = np.mean)
    CI_excl = summary_analyser.bootstrap(scores_excl_idk, bootstrap_stats_function = np.mean)
    
    print(f"\nRank Distribution of {model_identifier}")
    print(f"n_Perfect = {n_perfect}")
    print(f"n_mid = {n_mid}")
    print(f"n_bad = {n_bad}")
    print(f"n_IDK = {n_IDK}")
    print(f"Average excluding IDK = {avg_excluding_idk} with 95% CI [{CI_excl[0], CI_excl[1]}]")
    print(f"Average including IDK = {avg_including_idk} with 95% CI [{CI_incl[0], CI_incl[1]}]")


def present_hallucination_distribution(lst, model_identifier):
    n_IH = 0
    n_EH = 0

    for val in lst:
        if str(val) == "IH":
            n_IH += 1
        elif str(val) == "EH":
            n_EH += 1
    
    print(f"\nHallucination Distribution of {model_identifier}")
    print(f"N_Intrinsic_Hallucination = {n_IH}")
    print(f"N_Extrinsic_Hallucination = {n_EH}")
    














if __name__ == "__main__":
    main()