

import sys 

from tqdm import tqdm

from FileHandler import FileHandler

sys.path.append("..")

from Tuning import hyperparameters
from Utils.listutils import list_distinct


'''
    dataset: A dataset of the custom dataset class (must implement getName).
    qids: List of query ids.
'''
def get_retrieved_contexts(data_iden, qids, storage_iden):
    file_handler = FileHandler(storage_iden)

    data_set_name = data_iden.split("_")[0]

    if data_set_name == "Squad":
        (b, k1) = hyperparameters.WikiMultiHop_BM25_tuned()
    elif data_set_name == "WikiMultiHop":
        (b, k1) = hyperparameters.WikiMultiHop_BM25_tuned()
    else:
        raise Exception("Unknown Dataset name. Must be either Squad or WikiMultiHop")

    # A df with columns qid, docno, score, rank
    retrieval_df = file_handler.load_df(dir = "RetrievalFiles", dataset_name = data_iden, filename = "ModelFiles/RetrievalResults_b={}_k1={}.csv".format(str(b), str(k1)))
    # A df with columns docno, text
    docs_df = file_handler.load_df(dir = "DataFiles", dataset_name = data_iden, filename = "docs.csv")
    qrels_df = file_handler.load_df(dir = "DataFiles", dataset_name = data_iden, filename = "qrels.csv")

    assert(list(qrels_df["qid"]) == qids)

    # Construct mapping of docno to context
    docno_to_context = {}

    for (docno, context) in zip(list(docs_df["docno"]), list(docs_df["text"])):
        docno_to_context[docno] = context

    # ASSUMPTION: ASSUME K = 1
    qid_to_docno = {}

    retrieval_qids = list(retrieval_df["qid"])
    retrieval_docnos = list(retrieval_df["docno"])

    assert(len(retrieval_qids) == len(retrieval_docnos))

    # https://stackoverflow.com/questions/522372/finding-first-and-last-index-of-some-value-in-a-list-in-python
    print("Finding RAG docs ...")
    for qid in tqdm(list_distinct(retrieval_qids)):
        # Find first index of qids in retrieval_qids and use it find the first relevant document
        first_index = retrieval_qids.index(qid)
        # Find the docno that corresponds to the line
        qid_to_docno[qid] = retrieval_docnos[first_index]

    contexts = []
    RAG_docnos = []
    true_docnos = list(qrels_df["docno"])

    retrieved_qids = list_distinct(retrieval_qids)

    for qid in qids:
        if qid in retrieved_qids:
            docno = qid_to_docno[qid]
            retrieved_context = str(docno_to_context[docno])
            RAG_docnos.append(docno)
            contexts.append(retrieved_context)
        # For qids, which are not listed in the retrieval file.
        else:
            contexts.append("")
            RAG_docnos.append(-1)


    return list(contexts), list(RAG_docnos), true_docnos


'''
    TODO: Modify to also evaluate the RAG model.
'''
def runEval(data_iden, qids, storage_iden):
    RAG_contexts, RAG_docnos, true_docnos = get_retrieved_contexts(data_iden, qids, storage_iden)

    RAG_qids = []
    RAG_delete_qids = []

    for i,(RAG_docno, true_docno) in enumerate(zip(RAG_docnos, true_docnos)):
        if int(RAG_docno) != int(true_docno) and int(RAG_docno) != -1:
            RAG_qids.append(qids[i])
        elif int(RAG_docno) == -1:
            RAG_delete_qids.append(qids[i])

    return RAG_qids


def get_RAG_qids(storage_iden):
    file_handler = FileHandler(storage_iden)
    df_old = file_handler.load_df(dir = "DataFiles", dataset_name = "WikiMultiHop_Test", filename = "collab.csv")
    qids = list(df_old["qid"])
    RAG_qids = runEval("WikiMultiHop_Test", qids, storage_iden)

    return RAG_qids

def main():
    RAG_qids_old = get_RAG_qids("Storage - Old")
    RAG_qids_correct = get_RAG_qids("Storage - Correct")

    print(len(RAG_qids_old))
    print(len(RAG_qids_correct))

    qid_diff1 = []
    qid_diff2 = []

    for qid in RAG_qids_correct:
        if qid not in RAG_qids_old:
            qid_diff1.append(qid)

    for qid in RAG_qids_old:
        if qid not in RAG_qids_correct:
            qid_diff2.append(qid)
    
    print(qid_diff1)
    print(qid_diff2)
    

if __name__ == "__main__":
    main()