import sys

import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import random

from PytterierRetriever import BM25
from PytterierIndex import PytterierIndex

from tqdm import tqdm

import pandas as pd

sys.path.append("..")

from Tuning import hyperparameters

from Filehandling.FileHandler import FileHandler
from TextAnalysis.TextAnalysis import word_histogram

from transformers import AutoTokenizer
from Constants import constants

from LLM.SentenceBERT import SentenceBERT

# NOTE: Lot of code taken from RobustnessAnalysis.py

def main():
    num_args = len(sys.argv)

    if num_args != 2:
        raise Exception("Program must be run with exactly one argument, denoting which dataset to evaluate over")
    
    data_set_iden = sys.argv[1]

    file_handler = FileHandler()

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

    data_set_name = data_set_iden.split("_")[0]
    
    if data_set_name == "Squad":
        (b, k1) = hyperparameters.Squad_BM25_tuned()
    elif data_set_name == "WikiMultiHop":
        (b, k1) = hyperparameters.WikiMultiHop_BM25_tuned()
    else:
        raise Exception("Unknown Dataset name. Must be either Squad or WikiMultiHop")
    
    df_query = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "queries.csv")
    df_answer = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "answers.csv")
    df_qrels = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "qrels.csv")
    df_docs = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "docs.csv")

    joined = df_query.merge(df_answer, on="qid", how = "left", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
    joined = joined.merge(df_qrels, on="qid", how = "left", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
    df_data_files = joined.merge(df_docs, on="docno", how = "left", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

    df_retrieval = file_handler.load_df(dir = "RetrievalFiles", dataset_name = data_set_iden, filename = f"ModelFiles/RetrievalResults_b={b}_k1={k1}.csv")

    retrieval_qids = list(df_retrieval["qid"])
    retrieval_docnos = list(df_retrieval["docno"])


    qids = list(df_data_files["qid"])
    questions = list(df_data_files["query"])
    true_docnos = list(df_data_files["docno"])
    true_contexts = list(df_data_files["text"])

    docno_to_context = {}

    for (docno, context) in zip(list(df_docs["docno"]), list(df_docs["text"])):
        docno_to_context[docno] = context

    qids_wrong_retrieval = []
    questions_wrong_retrieval = []
    contexts_wrong_retrieval = []
    retrieval_scores_wrong = []

    qids_correct_retrieval = []
    questions_correct_retrieval = []
    contexts_correct_retrieval = []
    retrieval_scores_correct = []

    hard_high_attending_qids = []

    index = PytterierIndex(dataset_name = data_set_iden).load_index()
    bm25 = BM25(index = index, b = b, k1 = k1, num_results = 2500)

    print("Finding RAG docs ...")
    for (qid, question, true_docno) in tqdm(zip(qids, questions, true_docnos)):
        # Find first index of qids in retrieval_qids and use it find the first relevant document
        try:
            first_index = retrieval_qids.index(qid)
        except:
            print("Could not find qid = {} in retrieval results file".format(qid))
            continue
        
        retrieved_docno = retrieval_docnos[first_index]

        df = pd.DataFrame({"qid" : [qid], "query" : [question]})
        res = bm25.retrieve(df)

        try:
            ret_scores = list(res[res["docno"] == str(true_docno)]["score"])
            assert(len(ret_scores) == 1)
            ret_score = ret_scores[0]
        except:
            # Assume cases where relevant docno is not included in file is because the retieval score is 0.
            ret_score = 0


        if retrieved_docno != true_docno:
            qids_wrong_retrieval.append(qid)
            questions_wrong_retrieval.append(question)
            contexts_wrong_retrieval.append(docno_to_context[retrieved_docno])
            retrieval_scores_wrong.append(ret_score)

            if ret_score > 50.0:
                hard_high_attending_qids.append((qid,ret_score))

        else:
            qids_correct_retrieval.append(qid)
            questions_correct_retrieval.append(question)
            contexts_correct_retrieval.append(docno_to_context[true_docno])
            retrieval_scores_correct.append(ret_score)
                

    print("Hard high attending qids = {}".format(hard_high_attending_qids))
    random.seed(constants.random_seed())
    analysis(questions_correct_retrieval, contexts_correct_retrieval, retrieval_scores_correct,
              questions_wrong_retrieval, contexts_wrong_retrieval,  retrieval_scores_wrong,
              file_handler, data_set_iden, data_set_name, tokenizer)


def analysis(questions_correct, contexts_correct, ret_scores_correct, questions_wrong, contexts_wrong, ret_scores_wrong, file_handler, data_iden, data_name, tokenizer):
    save_folder = file_handler.storage_path() + f"/RetrievalFiles/{data_iden}/ErrorAnalysis/"

    # Boxplots over question and context lengths

    data_dict = {"Hard({})".format(len(questions_wrong)) : [len(tokenizer.tokenize(q)) for q in questions_wrong],
                  "Easy({})".format(len(questions_correct)) : [len(tokenizer.tokenize(q)) for q in questions_correct]}
    plt.title(f"{data_name}: Question Length", fontsize = 20)
    plt.boxplot(data_dict.values(), tick_labels=data_dict.keys())
    plt.ylabel("Question Length", fontsize = 20)
    plt.tick_params("x", labelsize = 20)
    plt.tick_params("y", labelsize = 20)
    plt.tight_layout()
    plt.savefig(save_folder + "Question_Length_Distribution.png")
    plt.close()

    data_dict = {"Hard({})".format(len(contexts_wrong)) : [len(tokenizer.tokenize(q)) for q in contexts_wrong],
                  "Easy({})".format(len(contexts_correct)) : [len(tokenizer.tokenize(q)) for q in contexts_correct]}
    plt.title(f"{data_name}: Context Length", fontsize = 20)
    plt.boxplot(data_dict.values(), tick_labels=data_dict.keys())
    plt.ylabel("Context Length", fontsize = 20)
    plt.tick_params("x", labelsize = 20)
    plt.tick_params("y", labelsize = 20)
    plt.tight_layout()
    plt.savefig(save_folder + "Context_Length_Distribution.png")
    plt.close()

    data_dict = {"Hard({})".format(len(ret_scores_wrong)) : ret_scores_wrong,
                  "Easy({})".format(len(ret_scores_correct)) : ret_scores_correct}
    plt.title(f"{data_name}: Retrieval Score", fontsize = 20)
    plt.boxplot(data_dict.values(), tick_labels=data_dict.keys())
    plt.ylabel("Retrieval Score", fontsize = 20)
    plt.tick_params("x", labelsize = 20)
    plt.tick_params("y", labelsize = 20)
    plt.tight_layout()
    plt.savefig(save_folder + "Retrieval_Score_Distribution.png")
    plt.close()


    # Histograms over words for bad/good questions/contexts:
    word_histogram(texts = questions_wrong, save_path = save_folder + "Bad_Questions_Hist.png",
                   title = "Histogram of words in Bad Queries", tokenizer = word_tokenize
                     , max_words = 10)
    
    word_histogram(texts = questions_correct, save_path = save_folder + "Correct_Questions_Hist.png",
                   title = "Histogram of words in Good Queries", tokenizer = word_tokenize
                     , max_words = 10)
    
    word_histogram(texts = contexts_wrong, save_path = save_folder + "Bad_Contexts_Hist.png",
                   title = "Histogram of words in Bad Contexts", tokenizer = word_tokenize
                     , max_words = 10)
    
    word_histogram(texts = contexts_correct, save_path = save_folder + "Correct_Contexts_Hist.png",
                   title = "Histogram of words in Good Contexts", tokenizer = word_tokenize
                     , max_words = 10)

    # Analysis over sentenceBERT representations:

    embedder = SentenceBERT()

    E_Q_wrong = embedder.embed(questions_wrong)
    E_C_wrong = embedder.embed(contexts_wrong)
    
    assert(len(E_Q_wrong) == len(E_C_wrong))

    E_D_wrong = [embedder.distance(E_Q_wrong[i, :], E_C_wrong[i, :]) for i in range(len(E_Q_wrong))]

    E_Q_correct = embedder.embed(questions_correct)
    E_C_correct = embedder.embed(contexts_correct)

    assert(len(E_Q_correct) == len(E_C_correct))

    E_D_correct = [embedder.distance(E_Q_correct[i, :], E_C_correct[i, :]) for i in range(len(E_Q_correct))]

    data_dict = {"Hard({})".format(len(contexts_wrong)) : E_D_wrong,
                  "Easy({})".format(len(contexts_correct)) : E_D_correct}
    plt.title(f"{data_name}: Cosine Distance", fontsize = 20)
    plt.boxplot(data_dict.values(), tick_labels=data_dict.keys())
    plt.tick_params("x", labelsize = 20)
    plt.tick_params("y", labelsize = 20)
    plt.ylabel("Cosine Distance", fontsize = 18)
    plt.tight_layout()
    plt.savefig(save_folder + "Similarity_Distance_Distribution.png")
    plt.close()

    # Take 30 bad and good samples:
    # https://note.nkmk.me/en/python-random-choice-sample-choices/

    n_sample = 30

    bad_questions = random.sample(questions_wrong, n_sample)
    good_questions = random.sample(questions_correct, n_sample)

    df_bad = pd.DataFrame({"Question" : bad_questions})
    df_good = pd.DataFrame({"Question" : good_questions})

    df_bad.to_latex(buf = file_handler.storage_path() + f"/RetrievalFiles/{data_iden}/ErrorAnalysis/wrong.tex",
                    escape = True)
    df_good.to_latex(buf = file_handler.storage_path() + f"/RetrievalFiles/{data_iden}/ErrorAnalysis/good.tex",
                    escape = True)


    
if __name__ == "__main__":
    main()