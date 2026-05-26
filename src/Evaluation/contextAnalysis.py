
import os 
import sys
import matplotlib.pyplot as plt
import numpy as np

import random

sys.path.append("..")

from Filehandling.FileHandler import FileHandler
from Constants import constants


# https://www.geeksforgeeks.org/python/how-to-set-the-x-and-the-y-limit-in-matplotlib-with-python/
def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # https://www.nielit.gov.in/gorakhpur/sites/default/files/Gorakhpur/ALevel_1_Python_20May.pdf
    num_args = len(sys.argv)

    if num_args != 3:
        raise Exception("Program must be run with exactly two arguments")
    
    data_set_iden = sys.argv[1]
    y_name = sys.argv[2]

    file_handler = FileHandler()

    df_qrels = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "qrels.csv")
    df_docs = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "docs.csv")
    df_M_context = file_handler.load_df(dir = "EvaluationFiles", dataset_name = data_set_iden, filename = "M_context.csv")

    # https://stackoverflow.com/questions/16476924/how-can-i-iterate-over-rows-in-a-pandas-dataframe

    # Construct dictionary that maps qid to its relevant evaluation score:
    qid_to_score = {}

    for idx, s in df_M_context.iterrows():
        qid_to_score[s["qid"]] = s[y_name]

    # Construct dictionary that maps docno to the set of qids where it serves as context as well as their evaluation scores.
    docno_to_qids_scores = {}

    for idx, s in df_qrels.iterrows():
        qid = int(s["qid"])
        docno = int(s["docno"])

        if docno in list(docno_to_qids_scores.keys()):
            lst_qids, lst_scores = docno_to_qids_scores[docno]

            lst_qids.append(qid)
            lst_scores.append(qid_to_score[qid])

            docno_to_qids_scores[docno] = (lst_qids, lst_scores)
        else:
            docno_to_qids_scores[docno] = [qid], [qid_to_score[qid]]
    
    # Map docno to context
    docno_to_context = {}

    for idx, s in df_docs.iterrows():
        docno_to_context[int(s["docno"])] = s["text"]

    context_docnos = [docno for docno in list(docno_to_qids_scores.keys())]
    context_values = [scores for qids, scores in list(docno_to_qids_scores.values())]

    context_means = [sum(lst) / len(lst) for lst in context_values]

    
    plt.title("Mean Context Score Distribution", fontsize = 18)
    plt.xlabel(f"Mean {y_name}", fontsize = 18)
    plt.ylabel("Frequency", fontsize = 18)
    plt.tick_params("x", labelsize = 18)
    plt.tick_params("y", labelsize = 18)
    plt.xlim(0.5, 1.05)
    plt.ylim(0, 80)
    plt.hist(x = context_means, bins = 100, color = "blue", label = "True Distribution")
    plt.tight_layout()
    plt.savefig(file_handler.storage_path() + f"/ContextAnalysis/{data_set_iden}/ContextDistribution.png")
    plt.close()

    arg_sort = np.argsort(np.array(context_means))

    bottom_3_context_docnos = [context_docnos[idx] for idx in arg_sort[0:3]]

    print(bottom_3_context_docnos)

    # Random Charts

    random.seed(constants.random_seed())
    scores = list(qid_to_score.values())

    color = "black"

    for i in range(0, 3):
        S = random_partition(scores)
        plt.title("Random Partition", fontsize = 18)
        plt.xlabel(f"Mean {y_name}", fontsize = 18)
        plt.ylabel("Frequency", fontsize = 18)
        plt.tick_params("x", labelsize = 18)
        plt.tick_params("y", labelsize = 18)
        plt.xlim(0.5, 1.05)
        plt.ylim(0, 80)
        plt.hist(x = S, bins = 100, color = color, alpha = 0.7, label = "sim_{}".format(i + 1))
        plt.legend()
        plt.tight_layout()
        plt.savefig(file_handler.storage_path() + f"/ContextAnalysis/{data_set_iden}/sim_{i}.png")
        plt.close()
    

def random_partition(scores):
    s = scores.copy()
    random.shuffle(s)

    B = int(len(scores) / constants.squad_sample_questions_per_contexts())

    S = [sum(s[i * constants.squad_sample_questions_per_contexts():(i + 1) * constants.squad_sample_questions_per_contexts()]
             ) / constants.squad_sample_questions_per_contexts() for i in range(B)]

    assert(len(S) == constants.squad_sample_contexts_size())
    return S


if __name__ == "__main__":
    main()