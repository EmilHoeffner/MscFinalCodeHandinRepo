# For all samples where retrieves non relevant doc, compare answer to M_question

# For all samples where M_question has EM = 1, fabricate fake context which contradicts and see what happens. 


import sys
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

sys.path.append("..")

from Filehandling.FileHandler import FileHandler

from Prompting.PromptLoader import PromptLoader
from LLM.Flan import FlanT5
from Tuning import hyperparameters
from LLM.SentenceBERT import SentenceBERT
from SummaryAnalysis.SummaryAnalyser import SummaryAnalyser

def main():
    # https://www.nielit.gov.in/gorakhpur/sites/default/files/Gorakhpur/ALevel_1_Python_20May.pdf
    num_args = len(sys.argv)

    if num_args != 4:
        raise Exception("Program must be run with exactly three arguments")
    
    data_set_iden = sys.argv[1]
    prompter_iden = sys.argv[2]
    BM25_mode = sys.argv[3]

    file_handler = FileHandler()

    df_query = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "queries.csv")
    df_answer = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "answers.csv")
    df_qrels = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "qrels.csv")
    df_docs = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "docs.csv")

    joined = df_query.merge(df_answer, on="qid", how = "left", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
    joined = joined.merge(df_qrels, on="qid", how = "left", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
    df_data_files = joined.merge(df_docs, on="docno", how = "left", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

    data_set_name = data_set_iden.split("_")[0]

    if BM25_mode == "Tune":
        (b, k1) = hyperparameters.BM25_default()
    elif BM25_mode == "Test":
        if data_set_name == "Squad":
            (b, k1) = hyperparameters.Squad_BM25_tuned()

        elif data_set_name == "WikiMultiHop":
            (b, k1) = hyperparameters.WikiMultiHop_BM25_tuned()
        else:
            raise Exception("Unknown Dataset name. Must be either Squad or WikiMultiHop")
    else:
        raise Exception("BM25 mode must either be Tune or Test")

    df_retrieval_results = file_handler.load_df(dir = "RetrievalFiles", dataset_name = data_set_iden, filename = f"ModelFiles/RetrievalResults_b={b}_k1={k1}.csv")

    df_results_question = file_handler.load_df(dir = "EvaluationFiles", dataset_name = data_set_iden, filename = "M_question.csv")
    df_results_RAG = file_handler.load_df(dir = "EvaluationFiles", dataset_name = data_set_iden, filename = "M_RAG.csv")

    print("\nIrrelevant Context Analysis\n")
    df = analysis_non_relevant_context(df_data_files, df_docs, df_retrieval_results, df_results_question, df_results_RAG, data_set_name)
    file_handler.save_df(df = df, dir = "RobustnessAnalysis", dataset_name = data_set_iden, filename = "NonRelevantContext.csv", header = True)

    print("\nFake Context Analysis\n")
    analysis_fake_context(prompter_iden, df_data_files, df_results_question, data_set_name)

# https://www.geeksforgeeks.org/pandas/ways-to-filter-pandas-dataframe-by-column-values/
def analysis_non_relevant_context(df_data_files, df_docs, df_retrieval, df_results_question, df_results_RAG, data_set_name):
    true_docnos = list(df_data_files["docno"])

    docno_to_context = {}

    for (docno, context) in zip(df_docs["docno"], df_docs["text"]):
        docno_to_context[docno] = context

    qids = list(df_data_files["qid"])
    questions = list(df_data_files["query"])

    qids_wrong_retrieval = []
    contexts_wrong_retrieval = []
    questions_wrong_retrieval = []
    scores_wrong_retrieval = []

    retrieval_qids = list(df_retrieval["qid"])
    retrieval_docnos = list(df_retrieval["docno"])
    retrieval_scores = list(df_retrieval["score"])

    # https://stackoverflow.com/questions/522372/finding-first-and-last-index-of-some-value-in-a-list-in-python
    print("Finding RAG docs ...")
    for (qid, question, true_docno) in zip(qids, questions, true_docnos):
        # Find first index of qids in retrieval_qids and use it find the first relevant document
        try:
            first_index = retrieval_qids.index(qid)
        except:
            print("Could not find qid = {} in retrieval results file".format(qid))
            continue
        
        retrieved_docno = retrieval_docnos[first_index]
        retrieved_score = retrieval_scores[first_index]

        if retrieved_docno != true_docno:
            qids_wrong_retrieval.append(qid)
            questions_wrong_retrieval.append(question)
            contexts_wrong_retrieval.append(docno_to_context[retrieved_docno])
            scores_wrong_retrieval.append(retrieved_score)

    print("Number of Wrong Retrievals = {}".format(len(qids_wrong_retrieval)))
    #print("Qids Wrong Retrieval = {}".format(qids_wrong_retrieval))
    
    M_question_answers = list( (df_results_question[df_results_question["qid"].isin(qids_wrong_retrieval)])["Predicted Answer"] )
    M_question_perplexities = list( (df_results_question[df_results_question["qid"].isin(qids_wrong_retrieval)])["Perp_ans"] )
    M_RAG_answers = list( (df_results_RAG[df_results_RAG["qid"].isin(qids_wrong_retrieval)])["Predicted Answer"] )
    true_answers = list( (df_results_question[df_results_question["qid"].isin(qids_wrong_retrieval)])["True Answer"] )

    M_RAG_f1 = list( (df_results_RAG[df_results_RAG["qid"].isin(qids_wrong_retrieval)])["F1"] )
    M_RAG_BERT = list( (df_results_RAG[df_results_RAG["qid"].isin(qids_wrong_retrieval)])["BERT"] )
    M_question_f1 = list( (df_results_question[df_results_question["qid"].isin(qids_wrong_retrieval)])["F1"] )

    taken_from_context = []
    RAG_idks = []
    question_idks = []
    unchanged = []

    taken_from_context_F1 = []
    taken_from_context_BERT = []

    taken_from_context_M_question_perp = []
    idk_M_question_perp = []
    unchanged_M_question_perp = []

    taken_from_context_retscore = []
    idk_retscore = []
    unchanged_retscore = []

    taken_from_context_semscore = []
    idk_semscore = []
    unchanged_semscore = []

    sentece_bert = SentenceBERT()

    print("Processing ...")
    for RAG_answer, question_answer, question_perp, ret_score, RAG_doc, question, F1, BERT in tqdm(zip(M_RAG_answers, M_question_answers, M_question_perplexities, retrieval_scores, contexts_wrong_retrieval, questions_wrong_retrieval, M_RAG_f1, M_RAG_BERT)):
        Q = sentece_bert.embed(question)
        D = sentece_bert.embed(RAG_doc)
        semantic_sim = sentece_bert.distance(Q, D)

        if str(RAG_answer) in str(RAG_doc):
            taken_from_context.append(1)
            taken_from_context_F1.append(F1)
            taken_from_context_BERT.append(BERT)
            taken_from_context_M_question_perp.append(question_perp)
            taken_from_context_retscore.append(ret_score)
            taken_from_context_semscore.append(semantic_sim)
        else:
            taken_from_context.append(0)

        if str(RAG_answer) == "IDK":
            RAG_idks.append(1)
            idk_M_question_perp.append(question_perp)
            idk_retscore.append(ret_score)
            idk_semscore.append(semantic_sim)
        else:
            RAG_idks.append(0)

        if str(question_answer) == "IDK":
            question_idks.append(1)
        else:
            question_idks.append(0)

        if str(RAG_answer) == str(question_answer):
            unchanged.append(1)
            unchanged_M_question_perp.append(question_perp)
            unchanged_retscore.append(ret_score)
            unchanged_semscore.append(semantic_sim)
        else:
            unchanged.append(0)

    df = pd.DataFrame({"qid" : qids_wrong_retrieval, "question" : questions_wrong_retrieval, "True Ans"  :  true_answers, "M_RAG" : M_RAG_answers, "IsSpan" : taken_from_context})

    summary_analyser = SummaryAnalyser(data_set_name)

    taken_from_context_CI = summary_analyser.bootstrap(x = taken_from_context, bootstrap_stats_function = np.mean,
                                                       alpha = 0.05)
    unchanged_CI = summary_analyser.bootstrap(x = unchanged, bootstrap_stats_function = np.mean,
                                                       alpha = 0.05)
    IDK_CI = summary_analyser.bootstrap(x = RAG_idks, bootstrap_stats_function = np.mean,
                                                       alpha = 0.05)
    

    print("Proportions of RAG answers taken verbatim from context is {} with 95% CI = [{},{}]".
          format(sum(taken_from_context)/len(M_RAG_answers), taken_from_context_CI[0], taken_from_context_CI[1]))
    
    print("Proportions of RAG Unchanged {} with 95% CI = [{},{}]".
          format(sum(unchanged)/len(M_RAG_answers), unchanged_CI[0], unchanged_CI[1]))


    print("\n")
    print("Proportions of RAG IDK {} with 95% CI = [{}, {}]".
          format(sum(RAG_idks)/len(M_RAG_answers), IDK_CI[0], IDK_CI[1]))

    print("Proportions of question IDK {}".format(sum(question_idks)/len(M_question_answers)))
    

    print("\n")

    print("M_RAG taken from context F1 = {}".format(sum(taken_from_context_F1) / len(taken_from_context_F1)))
    print("M_RAG taken from context BERT = {}".format(sum(taken_from_context_BERT) / len(taken_from_context_BERT)))

    # Box plot over perplexity scores of the 3 categories: Verbatim in context, IDK, Unchanged:

    box_plot_irrelevant_contexts(taken_from_context_M_question_perp, idk_M_question_perp, unchanged_M_question_perp, f"{data_set_name}: Distribution of Perplexity", "Perplexity")
    box_plot_irrelevant_contexts(taken_from_context_retscore, idk_retscore, unchanged_retscore, f"{data_set_name}: Distribution of RetrievalScore", "RetrievalScore")
    box_plot_irrelevant_contexts(taken_from_context_semscore, idk_semscore, unchanged_semscore, f"{data_set_name}: Distribution of Cosine Distance", "CosineDist(q,c)")

    return df

def box_plot_irrelevant_contexts(span, IDK, unchanged, title, measure_name):
    data_dict = {"Span({})".format(len(span)) : span,
                  "IDK({})".format(len(IDK)) : IDK, 
                  "=({})".format(len(unchanged)) : unchanged}
    plt.title(title, fontsize = 18)
    plt.boxplot(data_dict.values(), tick_labels=data_dict.keys())
    plt.ylabel(measure_name, fontsize = 18)
    plt.tick_params("x", labelsize = 18)
    plt.tick_params("y", labelsize = 18)
    plt.tight_layout()
    plt.show()
    plt.close()

def batch_predict(prompts, model):
    predicted_answers = []
    for prompt in prompts:
        predicted_answer, _ , _ = model.answer(prompt)
        predicted_answers.append(predicted_answer[0])

    return predicted_answers

    
def analysis_fake_context(prompter_iden, df_data_files, df_results_question, data_set_name): 
    n_sample = 50   
    df_question_filtered = df_results_question[df_results_question["EM"] == 1]
    # https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    df_question_filtered = df_question_filtered.sample(frac = 1, replace = False, random_state = 42)
    df_question_filtered = df_question_filtered[0:n_sample]

    qids_filtered = list(df_question_filtered["qid"])

    questions = list((df_data_files[df_data_files["qid"].isin(qids_filtered)])["query"])
    answers = list((df_data_files[df_data_files["qid"].isin(qids_filtered)])["answer"])
    true_contexts = list((df_data_files[df_data_files["qid"].isin(qids_filtered)])["text"])

    assert(len(questions) == n_sample)
    assert(len(true_contexts) == n_sample)

    file_handler = FileHandler()

    if not(file_handler.file_exists(dir = "RobustnessAnalysis", dataset_name = data_set_name + "_Test",
                                    filename = "CounterFactualSamples.csv")):
        file_handler.save_df(df = pd.DataFrame({"question" : questions, "answer" : answers, "CounterFactualContext" : ["NA" for _ in questions]}), 
                            dir = "RobustnessAnalysis", dataset_name = data_set_name + "_Test",
                            filename = "CounterFactualSamples.csv", header = True)
        raise Exception("Counter Factual File is not found. Saving one")
    else:
        df_count = file_handler.load_df(dir = "RobustnessAnalysis", dataset_name = data_set_name + "_Test",
                                    filename = "CounterFactualSamples.csv")
        if "NA" in list(df_count["CounterFactualContext"]):
            raise Exception("All Counter Factual Contexts has not yet been curated")

    counter_factual_contexts = list(df_count["CounterFactualContext"])

    prompter = PromptLoader().load_prompter(prompter_iden)

    prompts = prompter.construct_prompts_with_context(questions = questions, contexts = counter_factual_contexts)
    model = FlanT5()

    predicted_answers = batch_predict(prompts, model)

    results_with_default_prompt = pd.DataFrame({"question" : questions, "Old/True Answer" : answers, 
                                                "CounterFactualContext" : counter_factual_contexts,
                                                "New Answer" : predicted_answers})

    file_handler.save_df(df = results_with_default_prompt, dir = "RobustnessAnalysis", 
                         dataset_name = data_set_name + "_Test",
                         filename = "DefaultResults.csv", header = True)

    prompts = prompter.construct_safe_prompts_with_context(questions = questions, contexts = counter_factual_contexts)

    predicted_answers = batch_predict(prompts, model)

    results_with_safe_prompt = pd.DataFrame({"question" : questions, "Old/True Answer" : answers, 
                                                "CounterFactualContext" : counter_factual_contexts,
                                                "New Answer" : predicted_answers})

    file_handler.save_df(df = results_with_safe_prompt, dir = "RobustnessAnalysis", 
                         dataset_name = data_set_name + "_Test",
                         filename = "SafeResults.csv", header = True)


    prompts = prompter.construct_safe_prompts_with_context(questions = questions, contexts = true_contexts)

    predicted_answers = batch_predict(prompts, model)

    results_safe_reference = pd.DataFrame({"question" : questions, "Old/True Answer" : answers, 
                                                "CounterFactualContext" : counter_factual_contexts,
                                                "New Answer" : predicted_answers})

    file_handler.save_df(df = results_safe_reference, dir = "RobustnessAnalysis", 
                         dataset_name = data_set_name + "_Test",
                         filename = "SafeReferenceResults.csv", header = True)


if __name__ == "__main__":
    main()