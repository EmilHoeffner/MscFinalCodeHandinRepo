
from PytterierIndex import PytterierIndex
from PytterierRetriever import BM25
import sys

sys.path.append("..")

from Filehandling.FileHandler import FileHandler

from tqdm import tqdm

import time

# Computation time for Squad_Tune, with 11.000 samples was 8.3 minutes

def main():
    num_args = len(sys.argv)

    if num_args != 2:
        raise Exception("Program must be run with exactly one argument, denoting which dataset to evaluate over")
    
    data_set_iden = sys.argv[1]

    index = PytterierIndex(dataset_name = data_set_iden).load_index()
    file_handler = FileHandler()

    b_values = [0.1, 0.25, 0.5, 0.75, 0.9]
    k1_values = [1.0, 1.2, 1.4, 1.6, 1.8]

    time_start = time.time()

    for b in tqdm(b_values):
        for k1 in k1_values:
            # https://www.datacamp.com/tutorial/pandas-read-csv?utm_cid=23340058065&utm_aid=192632748929&utm_campaign=230119_1-ps-dscia~dsa-tofu~python_2-b2c_3-emea_4-prc_5-na_6-na_7-le_8-pdsh-go_9-nb-e_10-na_11-na&utm_loc=1005053-&utm_mtd=-c&utm_kw=&utm_source=google&utm_medium=paid_search&utm_content=ps-dscia~emea-en~dsa~tofu~tutorial~python&gad_source=1&gad_campaignid=23340058065&gclid=Cj0KCQiA4eHLBhCzARIsAJ2NZoKiBJ5yH0rN64N7-elnm-w7diQIXRFfhw6sHpoW5i8vmUWbI-BCnj4aAqf8EALw_wcB
            df_queries = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden , filename = "queries.csv")
            retriever = BM25(index, b = b, k1 = k1)
            res = retriever.retrieve(df_queries)
            file_handler.save_df(df = res, dir = "RetrievalFiles", dataset_name = data_set_iden , filename = f"ModelFiles/RetrievalResults_b={b}_k1={k1}.csv", header = True)

    time_end = time.time()
    total_time = (time_end - time_start) / 60.0 

    print("Total time for retrieval: {} minutes".format(total_time))


if __name__ == "__main__":
    main()