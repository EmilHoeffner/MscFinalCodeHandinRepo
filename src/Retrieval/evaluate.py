
import sys

from os import listdir
from os.path import isfile, join
import time
from PytterierEvaluater import PytterierEvaluater
import pandas as pd

sys.path.append("..")

from Filehandling.FileHandler import FileHandler

# Computation time for Squad_Tune, with 11.000 samples was 0.58 minutes

def main():
    num_args = len(sys.argv)

    if num_args != 2:
        raise Exception("Program must be run with exactly one argument, denoting which dataset to evaluate over")
    
    data_set_iden = sys.argv[1]

    file_handler = FileHandler()

    results_dir = file_handler.storage_path() + f"/RetrievalFiles/{data_set_iden}/ModelFiles/"

    # https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    results_files = [f for f in listdir(results_dir) if isfile(join(results_dir, f)) and f != ".gitkeep"]
    names = [pretty_name(f) for f in results_files]

    results_dfs = [file_handler.load_df(dir = "RetrievalFiles", dataset_name = data_set_iden, filename = f"ModelFiles/{f}") for f in results_files]

    queries = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "queries.csv")
    qrels = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "qrels.csv")
    
    evaluater = PytterierEvaluater()

    time_start = time.time() 
    results = evaluater.evaluate(dfs = results_dfs, names = names, queries = queries, qrels = qrels)
    file_handler.save_df(df = results, dir = "RetrievalFiles", dataset_name = data_set_iden, filename = f"EvaluationFile/Eval.csv", header = True)
    time_end = time.time() 

    time_total = (time_end - time_start) / 60.0 

    print("Time to evaluate = {}".format(time_total))



def pretty_name(file_name):
    s = file_name.split("_")
    b_value = float(s[1].split("=")[1])
    k1_value = float(s[2].split(".csv")[0].split("=")[1])
    return f"BM25(b = {b_value},k1 = {k1_value})"

    
if __name__ == "__main__":
    main()