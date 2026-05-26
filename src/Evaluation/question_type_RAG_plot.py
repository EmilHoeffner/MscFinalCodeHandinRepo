

import sys 
import os
import matplotlib.pyplot as plt

sys.path.append("..")

from Filehandling.FileHandler import FileHandler
from Utils.listutils import list_distinct


def histogram(vals, y_name, question_type):
    plt.title(f"{question_type}", fontsize = 18)
    plt.hist(vals, bins = 20)
    plt.xlabel(y_name, fontsize = 18)
    plt.ylabel("Frequency", fontsize = 18)
    plt.tick_params("x", labelsize = 18)
    plt.tick_params("y", labelsize = 18)
    plt.tight_layout()
    plt.show()
    plt.close()

def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # https://www.nielit.gov.in/gorakhpur/sites/default/files/Gorakhpur/ALevel_1_Python_20May.pdf
    num_args = len(sys.argv)

    if num_args != 2:
        raise Exception("Program must be run with exactly one argument")
    
    data_set_iden = "WikiMultiHop_Test"
    y_name = sys.argv[1]

    file_handler = FileHandler()

    df_queries = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "queries.csv")
    df_results = file_handler.load_df(dir = "EvaluationFiles", dataset_name = data_set_iden, filename = "M_RAG.csv")

    joined = df_queries.merge(df_results, on="qid", how = "left", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

    question_types = list(joined["type"])
    scores = joined[y_name]

    for question_type in list_distinct(question_types):
        indexes = [i for i in range(len(question_types)) if question_types[i] == question_type]
        scores_qt = [score for i,score in enumerate(scores) if i in indexes]

        histogram(scores_qt, y_name, question_type)














if __name__ == "__main__":
    main()