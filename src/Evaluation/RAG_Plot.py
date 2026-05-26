import os
import sys

import matplotlib.pyplot as plt

sys.path.append("..")

from evaluate import get_retrieved_contexts
from scipy.stats import pointbiserialr

from Filehandling.FileHandler import FileHandler

from Dataloading.Dataloader import DataLoader

def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # https://www.nielit.gov.in/gorakhpur/sites/default/files/Gorakhpur/ALevel_1_Python_20May.pdf
    num_args = len(sys.argv)

    if num_args != 3:
        raise Exception("Program must be run with exactly two arguments")
    
    data_set_iden = sys.argv[1]
    y_name = sys.argv[2]

    file_handler = FileHandler()

    dl = DataLoader()
    dataset = dl.load_data(data_set_iden)

    qids = dataset.get_qids()
    _, RAG_docnos, _ = get_retrieved_contexts(data_set_iden, qids, tuned = True)

    df_qrels = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "qrels.csv")
    assert(list(df_qrels["qid"]) == list(qids))

    true_docnos = list(df_qrels["docno"])

    df_M_RAG = file_handler.load_df(dir = "EvaluationFiles",  dataset_name = data_set_iden ,filename = "M_RAG.csv")
    df_M_question = file_handler.load_df(dir = "EvaluationFiles",  dataset_name = data_set_iden ,filename = "M_question.csv")

    RAG_scores = list(df_M_RAG[y_name]) 
    question_scores = list(df_M_question[y_name])

    RAG_scores_relevant_docs = []
    RAG_scores_nonrelevant_docs = []
    question_scores_non_relevant = []

    for i, score in enumerate(RAG_scores):
        if int(RAG_docnos[i]) == int(true_docnos[i]):
            RAG_scores_relevant_docs.append(score)
        else:
            RAG_scores_nonrelevant_docs.append(score)
            question_scores_non_relevant.append(question_scores[i])
            

    # https://www.geeksforgeeks.org/data-visualization/creating-multiple-boxplots-on-the-same-graph-from-a-dictionary/
    # Box Plot

    X = [0 for _ in RAG_scores_nonrelevant_docs] + [1 for _ in RAG_scores_relevant_docs]
    Y = [s for s in RAG_scores_nonrelevant_docs] + [s for s in RAG_scores_relevant_docs]

    assert(len(X) == len(Y))

    cor = pointbiserialr(x = X, y = Y) 
    val = cor[0]
    p_val = cor[1]

    '''
    data_dict = {"R_Doc({})".format(len(RAG_scores_relevant_docs)) : RAG_scores_relevant_docs, 
                 "NR_Doc({})".format(len(RAG_scores_nonrelevant_docs)) : RAG_scores_nonrelevant_docs}
    plt.title(f"cor = {round(val, 3)} with p value {round(p_val, 3)}, R^2 = {round(val * val, 3)}")
    plt.boxplot(data_dict.values(), tick_labels=data_dict.keys())
    plt.ylabel(y_name)
    plt.show()
    plt.close()
    '''

    data_set_name = data_set_iden.split("_")[0]

    stats_str = f"cor = {round(val, 2)} with p value {round(p_val, 6)}, R^2 = {round(val * val, 2)}"

    plt.title(f"{data_set_name}: distribution of {y_name} for X = 1", fontsize = 16)
    plt.hist(x = RAG_scores_relevant_docs, bins = 20)
    plt.xlabel(y_name, fontsize = 16)
    plt.ylabel("Frequency", fontsize = 16)
    plt.tick_params("x", labelsize = 16)
    plt.tick_params("y", labelsize = 16)
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.title(f"{data_set_name}: distribution of {y_name} for X = 0", fontsize = 16)
    plt.hist(x = RAG_scores_nonrelevant_docs, bins = 20)
    plt.xlabel(y_name, fontsize = 16)
    plt.ylabel("Frequency", fontsize = 16)
    plt.tick_params("x", labelsize = 16)
    plt.tick_params("y", labelsize = 16)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    print(stats_str)


if __name__ == "__main__":
    main()