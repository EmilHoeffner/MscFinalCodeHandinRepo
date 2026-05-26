from PytterierIndex import PytterierIndex
from PytterierRetriever import BM25
import sys

sys.path.append("..")

from Filehandling.FileHandler import FileHandler
from Dataloading.Dataloader import DataLoader

from tqdm import tqdm

import time
from Tuning import hyperparameters

# Computation time for Squad_Tune, with 11.000 samples was 8.3 minutes

def main():
    num_args = len(sys.argv)

    if num_args != 2:
        raise Exception("Program must be run with exactly three argument, denoting which dataset to evaluate over")
    
    data_set_iden = sys.argv[1]
    
    if data_set_iden == "Squad_Test":
        (b, k1) = hyperparameters.Squad_BM25_tuned()
    elif data_set_iden == "WikiMultiHop_Test":
        (b, k1) = hyperparameters.WikiMultiHop_BM25_tuned()
    else:
        raise Exception("Invalid Data set identifier")

    index = PytterierIndex(dataset_name = data_set_iden).load_index()
    file_handler = FileHandler()

    time_start = time.time()

    df_queries = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden , filename = "queries.csv")
    retriever = BM25(index, b = b, k1 = k1)
    res = retriever.retrieve(df_queries)
    file_handler.save_df(df = res, dir = "RetrievalFiles", dataset_name = data_set_iden , filename = f"ModelFiles/RetrievalResults_b={b}_k1={k1}.csv", header = True)
            
    time_end = time.time()
    total_time = (time_end - time_start) / 60.0 

    print("Total time for retrieval: {} minutes".format(total_time))


if __name__ == "__main__":
    main()