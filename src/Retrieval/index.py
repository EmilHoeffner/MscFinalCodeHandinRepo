
from PytterierIndex import PytterierIndex
import sys 
import pandas as pd

import ast

sys.path.append("..")

from Dataloading.Dataloader import DataLoader
from Filehandling.FileHandler import FileHandler

import time
import csv
from tqdm import tqdm

from Utils.listutils import list_distinct

# Computation time for Squad_Tune, with 11.000 samples, when "extended_qrels" is omitted, took 0.58 minutes.

# https://stackoverflow.com/questions/70902380/how-can-i-write-each-row-to-a-csv-file
# https://medium.com/@AlexanderObregon/how-to-read-and-write-csv-files-in-python-b84fe274d51a
def save_extended_qrels(qids, docnos, path):
    unique_docnos = list_distinct(docnos)

    header = ["qid", "docno", "label"]

    with open(path, "w", newline = "") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        N = len(qids)
        for i in tqdm(range(N)):
            rows = []
            qid = qids[i]
            for docno in unique_docnos:
                qid = str(qid)
                docno = str(docno)
                if docno == str(docnos[i]):
                    label = 1
                else:
                    label = 0

                rows.append([qid, docno, label])

            writer.writerows(rows)
                
def main():
    num_args = len(sys.argv)

    if num_args != 2:
        raise Exception("Program must be run with exactly one argument, denoting which dataset to evaluate over")
    
    data_set_iden = sys.argv[1]

    dl = DataLoader()
    data = dl.load_data(data_set_iden)

    index = PytterierIndex(dataset_name = data.get_name())

    contexts = data.get_contexts()
    questions = data.get_questions()
    answers = data.get_answers()
    answer_starts = data.get_answer_starts()
    answer_starts_relative = data.get_relative_answer_starts()
    question_types = data.get_question_types()

    file_handler = FileHandler()

    qids = data.get_qids()
    queries = [str(question) for question in questions]
    # https://www.geeksforgeeks.org/python/create-a-pandas-dataframe-from-lists/
    df_query = pd.DataFrame({"qid" : qids, "query" : queries, "type" : question_types})
    df_answer = pd.DataFrame({"qid" : qids, "answer" : answers, "answer_start" : answer_starts, "answer_start_relative" : answer_starts_relative})
    

    if data_set_iden == "WikiMultiHop_Test":
        support_sentences = data.get_supporting_sentences()

        S = []

        bad_qids = []
        
        for i,(sample,q_type) in enumerate(zip(support_sentences, question_types)):
            s = ""
            num_sent = 0
            for j,sent in enumerate(sample):
                if len(sent) == 1 and i != len(sample) - 1:
                    s += sent[0] + " "
                    num_sent += 1
                elif len(sent) == 1 and i == len(sample) - 1:
                    s += sent[0]
                    num_sent += 1

            if (q_type == "bridge_comparison" and num_sent != 4) or (q_type != "bridge_comparison" and num_sent != 2):
                bad_qids.append(i)

            S.append(s)
        
        print(f"Number of bad qids = {len(bad_qids)}")
        
        df_collab = pd.DataFrame({"qid" : qids, "question" : queries, "context" : contexts, "RelevantSentences" : S, "answer" : answers})
        df_bad_qids = pd.DataFrame({"qid" : bad_qids})
        file_handler.save_df(df = df_bad_qids, dir = "DataFiles", dataset_name = data.get_name(), filename = "BadQids.csv", header = True)
    else:
        df_collab = pd.DataFrame({"qid" : qids, "question" : queries, "context" : contexts, "answer" : answers})
    # https://stackoverflow.com/questions/17098654/how-to-reversibly-store-and-load-a-pandas-dataframe-to-from-disk
    # https://stackoverflow.com/questions/48053207/writing-single-csv-header-with-pandas
    file_handler.save_df(df = df_query, dir = "DataFiles", dataset_name = data.get_name(), filename = "queries.csv", header = True)
    file_handler.save_df(df = df_answer, dir = "DataFiles", dataset_name = data.get_name(), filename = "answers.csv", header = True)
    file_handler.save_df(df = df_collab, dir = "DataFiles", dataset_name = data.get_name(), filename = "collab.csv", header = True)

    questions_docid = []
    contexts_set_doc_id = {}

    cur_doc_id = 0
    for context in contexts:
        if context in list(contexts_set_doc_id.keys()):
            questions_docid.append(contexts_set_doc_id[context])
        else:
            contexts_set_doc_id[context] = cur_doc_id 
            questions_docid.append(cur_doc_id)
            cur_doc_id += 1

    unique_contexts = list_distinct(data.get_corpus())

    for c in unique_contexts:
        if c in list(contexts_set_doc_id.keys()):
            continue 
        
        contexts_set_doc_id[c] = cur_doc_id
        cur_doc_id += 1

    labels = [1 for qid in qids]

    df_qrels = pd.DataFrame({"qid" : qids, "docno" : questions_docid, "label" : labels})
    file_handler.save_df(df = df_qrels, dir = "DataFiles", dataset_name = data.get_name(), filename = "qrels.csv", header = True)


    if data_set_iden == "Squad_Dummy" or data_set_iden == "Squad_Tune" or data_set_iden == "Squad_Test":
        df_docs = pd.DataFrame({"docno" : contexts_set_doc_id.values(), "text" : contexts_set_doc_id.keys()})
    elif data_set_iden == "WikiMultiHop_Dummy" or data_set_iden == "WikiMultiHop_Tune" or data_set_iden == "WikiMultiHop_Test":
        supporting_contexts = data.get_supporting_contexts()
        support_sentences = data.get_supporting_sentences()
        distractive_contexts = data.get_distractive_contexts()
        df_docs =  pd.DataFrame({"docno" : contexts_set_doc_id.values(), "text" : contexts_set_doc_id.keys(),
                                 "support" : supporting_contexts, "support_sentences" : support_sentences, "distractive" : distractive_contexts})
    else:
        raise Exception("Wrong Dataset Identifier")
    
    
    file_handler.save_df(df = df_docs, dir = "DataFiles", dataset_name = data.get_name(), filename = "docs.csv", header = True)

    data = [{"docno" : str(contexts_set_doc_id[context]), "text" : str(context)} for context in unique_contexts]
    
    time_start = time.time()
    index.store_index(data)
    time_end = time.time()

    time_total = (time_end - time_start) / 60.0

    print("Run Time = {} minutes".format(time_total))


if __name__ == "__main__":
    main()




