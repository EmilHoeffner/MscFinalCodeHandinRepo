'''
    Some code has been taken from the Msc Thesis Code Repo
'''

import sys
from Evaluater import Evaluater
import os
import pandas as pd

# https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder
sys.path.append("..")

from Prompting.PromptLoader import PromptLoader
from Dataloading.Dataloader import DataLoader
from LLM.Flan import FlanT5
from Filehandling.FileHandler import FileHandler
from tqdm import tqdm
from Tuning import hyperparameters
from Constants import constants
from Utils.listutils import list_distinct

def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # https://www.nielit.gov.in/gorakhpur/sites/default/files/Gorakhpur/ALevel_1_Python_20May.pdf
    num_args = len(sys.argv)

    if num_args != 4:
        raise Exception("Program must be run with exactly three arguments")
    
    data_set_iden = sys.argv[1]
    prompter_iden = sys.argv[2]
    pruned = sys.argv[3]

    if pruned == "yes":
        print("Running pruned code")
        runEval(data_set_iden, prompter_iden, True)
    else:
        print("Running non-pruned code")
        runEval(data_set_iden, prompter_iden, False)

    

def runEval(dataset_iden, prompter_iden, pruned):
    file_handler = FileHandler()
    evaluater = Evaluater()
    model = FlanT5()
    prompt_engineer = PromptLoader().load_prompter(prompter_iden)

    dl = DataLoader()
    dataset = dl.load_data(dataset_iden)

    true_answers = dataset.get_answers()
    qids = dataset.get_qids()
    questions = dataset.get_questions()

    if pruned:
        collab_df = file_handler.load_df(dir = "DataFiles", dataset_name = dataset_iden, filename = "collab.csv")
        true_contexts = list(collab_df["RelevantSentences"])
        prompts_M_context = prompt_engineer.construct_prompts_with_context(questions, true_contexts)
        results_M_context = evaluater.evaluate(model, prompts_M_context, true_answers, qids)
        file_handler.save_df(df = results_M_context, dir = "EvaluationFiles", dataset_name = dataset.get_name() , filename = "M_context_pruned.csv", header = True)
        return 
    else:
        true_contexts = dataset.get_contexts()

    RAG_contexts, RAG_docnos, true_docnos = get_retrieved_contexts(dataset_iden, qids)
    
    RAG_qids = []
    RAG_delete_qids = []

    for i,(RAG_docno, true_docno) in enumerate(zip(RAG_docnos, true_docnos)):
        if int(RAG_docno) != int(true_docno) and int(RAG_docno) != -1:
            RAG_qids.append(qids[i])
        elif int(RAG_docno) == -1:
            RAG_delete_qids.append(qids[i])
    
    
    print("Number of faulty retrieved documents is {}".format(len(RAG_qids)))
    print("Number of Empty retrievals is {}".format(len(RAG_delete_qids)))
    print("Qids of Faulty Retrieved is {}".format(RAG_delete_qids))

    qids_RAG = [qid for i,qid in enumerate(qids) if i in RAG_qids]
    questions_RAG = [question for i,question in enumerate(questions) if i in RAG_qids]
    contexts_RAG = [context for i,context in enumerate(RAG_contexts) if i in RAG_qids]
    true_answers_RAG = [answer for i,answer in enumerate(true_answers) if i in RAG_qids]


    prompts_M_question = prompt_engineer.construct_prompts_without_context(questions)
    prompts_M_RAG = prompt_engineer.construct_prompts_with_context(questions_RAG, contexts_RAG)
    prompts_M_context = prompt_engineer.construct_prompts_with_context(questions, true_contexts)

    
    print(len(prompts_M_RAG))

    assert(len(prompts_M_question) == len(prompts_M_context))
    assert(len(prompts_M_context) != len(prompts_M_RAG))

    print(prompts_M_RAG[0])
    

    # https://saturncloud.io/blog/how-to-write-a-pandas-dataframe-to-a-txt-file/
    results_M_question = evaluater.evaluate(model, prompts_M_question, true_answers, qids)
    file_handler.save_df(df = results_M_question, dir = "EvaluationFiles",  dataset_name = dataset.get_name() ,filename = "M_question.csv", header = True)

    results_M_context = evaluater.evaluate(model, prompts_M_context, true_answers, qids)
    file_handler.save_df(df = results_M_context, dir = "EvaluationFiles", dataset_name = dataset.get_name() , filename = "M_context.csv", header = True)

    

    assert(len(prompts_M_RAG) == len(true_answers_RAG))
    assert(len(true_answers_RAG) == len(qids_RAG))
    results_M_RAG = evaluater.evaluate(model, prompts_M_RAG, true_answers_RAG, qids_RAG)
    # https://stackoverflow.com/questions/53727153/replacing-rows-in-pandas-dataframe-with-other-dataframe-based-on-index
    # https://stackoverflow.com/questions/39267372/replace-rows-in-a-pandas-df-with-rows-from-another-df
    # Replace rows in df_context, with rows of M_RAG, where the fetched document is wrong. 
    #cols = list(results_M_context.columns) 
    #results_M_context.loc[results_M_context.qid.isin(results_M_RAG.qid), cols] = results_M_RAG[cols]
    # https://stackoverflow.com/questions/22430195/pandas-dataframe-set-rows-equal-to-other-rows

    results_M_context[results_M_context["qid"].isin(qids_RAG)] = results_M_RAG.values
    # https://medium.com/@amit25173/filtering-data-in-pandas-basics-you-need-to-know-639ed999821b
    results_M_context = results_M_context[results_M_context["qid"] not in RAG_delete_qids]

    file_handler.save_df(df = results_M_context, dir = "EvaluationFiles",  dataset_name = dataset.get_name() ,filename = "M_RAG.csv", header = True)

    

    

'''
    dataset: A dataset of the custom dataset class (must implement getName).
    qids: List of query ids.
'''
def get_retrieved_contexts(data_iden, qids, tuned = False):
    file_handler = FileHandler()

    data_set_name = data_iden.split("_")[0]
    
    if data_set_name == "Squad":
        if tuned:
            (b, k1) = hyperparameters.Squad_BM25_tuned()
        else:
            (b, k1) = hyperparameters.BM25_default()
    elif data_set_name == "WikiMultiHop":
        if tuned:
            (b, k1) = hyperparameters.WikiMultiHop_BM25_tuned()
        else:
            (b, k1) = hyperparameters.BM25_default()
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

    # https://stackoverflow.com/questions/522372/finding-first-and-last-index-of-some-value-in-a-list-in-python
    print("Finding RAG docs ...")
    for qid in tqdm(list_distinct(retrieval_qids)):
        # Find first index of qids in retrieval_qids and use it find the first relevant document
        first_index = retrieval_qids.index(qid)
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

if __name__ == "__main__":
    main()