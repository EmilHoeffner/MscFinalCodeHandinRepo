
import sys 
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import entropy

from transformers import AutoTokenizer
from utils import overlap_indexes
import matplotlib.pyplot as plt
import ast
from scipy.stats import pointbiserialr

sys.path.append("..")

from Filehandling.FileHandler import FileHandler
from Prompting.PromptLoader import PromptLoader
from LLM.Flan import FlanT5
from SummaryAnalysis.SummaryAnalyser import SummaryAnalyser

from Constants import constants



def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # https://www.nielit.gov.in/gorakhpur/sites/default/files/Gorakhpur/ALevel_1_Python_20May.pdf
    num_args = len(sys.argv)

    if num_args != 3 and num_args != 4:
        raise Exception("Program must be run with two or three arguments")
    
    data_set_iden = sys.argv[1]
    prompter_iden = sys.argv[2]

    summary_analyser = SummaryAnalyser(data_set_iden)

    file_handler = FileHandler()
    prompter = PromptLoader().load_prompter(prompter_iden)

    df_M_context = file_handler.load_df(dir = "EvaluationFiles", dataset_name = data_set_iden, filename = "M_context.csv")

    df_docs = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "docs.csv")
    
    df_question = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "queries.csv")
    df_qrels = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "qrels.csv")
    

    df_combined = df_M_context.merge(df_question, on="qid", how = "left", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
    df_combined = df_combined.merge(df_qrels, on="qid", how = "left", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
    df_combined = df_combined.merge(df_docs, on="docno", how = "left", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

    if num_args == 4:
        question_type = str(sys.argv[3])
        print(f"Only over {question_type} questions")
        df_combined = df_combined[df_combined["type"].isin([question_type])]

    # https://stackoverflow.com/questions/13611065/efficient-way-to-apply-multiple-filters-to-pandas-dataframe-or-series
    combined_good = df_combined[df_combined["F1"] == 1.0]
    combined_bad = df_combined[(df_combined["F1"] == 0.0)]

    # https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html
    combined_good = combined_good.sample(frac = 1, replace = False, random_state = constants.random_seed()).reset_index(drop=True)
    combined_bad = combined_bad.sample(frac = 1, replace = False, random_state = constants.random_seed()).reset_index(drop=True)

    max_samples = 100
    combined_good = combined_good[0:max_samples]
    combined_bad = combined_bad[0:max_samples]


    prompts_good = to_prompts(combined_good, prompter)
    prompts_bad = to_prompts(combined_bad, prompter)

    model = FlanT5()
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl", device_map = "cuda")
    tokenizer.model_max_length = 4000
    
    data_set_name = data_set_iden.split("_")[0]
    
    if data_set_name == "Squad":
        # https://stackoverflow.com/questions/11285613/selecting-multiple-columns-in-a-pandas-dataframe
        combined_good = combined_good[["qid" , "query","True Answer", "Predicted Answer", "text"]]
        combined_bad = combined_bad[["qid", "query" ,"True Answer", "Predicted Answer", "text"]]

        good_queries = list(combined_good["query"])
        bad_queries = list(combined_bad["query"])
        good_answers = list(combined_good["True Answer"])
        bad_answers = list(combined_bad["True Answer"])

        golden_contexts_good = list(combined_good["True Answer"])
        golden_contexts_good_tokens = [tokenizer(str(a))["input_ids"] for a in golden_contexts_good]
        # Excluding last answer tokens, as everyone of them ends with id 1 (probably an end token)
        golden_contexts_good_tokens = [a[0:len(a)-1] for a in golden_contexts_good_tokens]
        prompts_good_tokens = [tokenizer(a)["input_ids"] for a in prompts_good]
        gold_positions_good = [overlap_indexes(p, a) for (a,p) in zip(golden_contexts_good_tokens, prompts_good_tokens)]
                
        golden_contexts_bad = list(combined_bad["True Answer"])
        golden_contexts_bad_tokens = [tokenizer(str(a))["input_ids"] for a in golden_contexts_bad]
        # Excluding last answer tokens, as everyone of them ends with id 1 (probably an end token)
        golden_contexts_bad_tokens = [a[0:len(a)-1] for a in golden_contexts_bad_tokens]
        prompts_bad_tokens = [tokenizer(a)["input_ids"] for a in prompts_bad]
        gold_positions_bad = [overlap_indexes(p, a) for (a,p) in zip(golden_contexts_bad_tokens, prompts_bad_tokens)]
        
    elif data_set_name == "WikiMultiHop":
        # https://stackoverflow.com/questions/11285613/selecting-multiple-columns-in-a-pandas-dataframe
        combined_good = combined_good[["qid" , "query","True Answer", "Predicted Answer", "text", "support", "support_sentences"]]
        combined_bad = combined_bad[["qid", "query" ,"True Answer", "Predicted Answer", "text", "support", "support_sentences"]]

        good_queries = list(combined_good["query"])
        bad_queries = list(combined_bad["query"])
        good_answers = list(combined_good["True Answer"])
        bad_answers = list(combined_bad["True Answer"])

        # https://bluevps.com/blog/how-to-convert-a-string-to-a-list-in-python-step-by-step-guide
        # https://codemia.io/knowledge-hub/path/how_to_convert_string_representation_of_list_to_a_list
        golden_supports_good = [ast.literal_eval(s) for s in list(combined_good["support_sentences"])]
        gold_positions_good = []
        prompts_good_tokens = [tokenizer(a)["input_ids"] for a in prompts_good]


        for i,lst in enumerate(golden_supports_good):
            gold_positions = []
            for sup in lst:
                for p in sup:
                    tokens = tokenizer(p)["input_ids"]
                    tokens = tokens[0:len(tokens)-1]
                    o = overlap_indexes(prompts_good_tokens[i], tokens)

                    if len(o) != len(tokens):
                        print("o = {}".format(o))
                        print("tokens = {}".format(tokens))
                        raise Exception("OUT")
                        
                    gold_positions += o
            gold_positions_good.append(gold_positions)
                

        golden_supports_bad = [ast.literal_eval(s) for s in list(combined_bad["support_sentences"])]
        gold_positions_bad = []
        prompts_bad_tokens = [tokenizer(a)["input_ids"] for a in prompts_bad]


        for i,lst in enumerate(golden_supports_bad):
            gold_positions = []
            for sup in lst:
                for p in sup:
                    tokens = tokenizer(p)["input_ids"]
                    tokens = tokens[0:len(tokens)-1]
                    o = overlap_indexes(prompts_bad_tokens[i], tokens)

                    if len(o) != len(tokens):
                        print("Outlier")
                        
                    gold_positions += o

            gold_positions_bad.append(gold_positions)
    else:
        raise Exception("Unknown Dataset name. Must be either Squad or WikiMultiHop")


    print("Computing Attention Maps for Bad Samples = {}".format(len(prompts_bad)))
    answers_bad, cross_attentions_bad, input_tokens_bad = model.cross_attention_map(prompts_bad)
    print("Computing Attention Maps for Good Samples: Size = {}".format(len(prompts_good)))
    answers_good, cross_attentions_good, input_tokens_good = model.cross_attention_map(prompts_good)
    

    for a1, a2 in zip(answers_good, combined_good["Predicted Answer"]):
        try:
            assert(str(a1) == str(a2))
        except:
            print(f"These were different:\n{a1}\n{a2}")
            continue

    for a1, a2 in zip(answers_bad, combined_bad["Predicted Answer"]):
        try:
            assert(str(a1) == str(a2))
        except:
            print(f"These were different:\n{a1}\n{a2}")
            continue

    
    print("Attention Analysis Bad:")
    top_tokens_bad, top_indexes_bad, explained_attention_bad = attention_analysis(cross_attentions_bad, input_tokens_bad)
    print("Attention Analysis Good:")
    top_tokens_good, top_indexes_good, explained_attention_good = attention_analysis(cross_attentions_good, input_tokens_good)
    

    recall_mean_bad = []
    recall_CI_bad = []
    precision_mean_bad = []
    precision_CI_bad = []
    explained_attention_mean_bad = []
    explained_attention_CI_bad = []

    recall_mean_good = []
    recall_CI_good = []
    precision_mean_good = []
    precision_CI_good = []
    explained_attention_mean_good = []
    explained_attention_CI_good = []

    correlations_F = []
    correlations_recall = []
    correlations_precision = []

    w_values = list(range(1, 51))

    print("Computing Plot Statistics:")
    for w in tqdm(w_values):
        TI_good = [t[0:w] for t in top_indexes_good]
        TI_bad = [t[0:w] for t in top_indexes_bad]

        recalls_good, mean_recall_good, CI_good = compute_recall(TI_good, gold_positions_good, summary_analyser)
        recalls_bad, mean_recall_bad, CI_bad = compute_recall(TI_bad, gold_positions_bad, summary_analyser)

        recall_mean_bad.append(mean_recall_bad)
        recall_CI_bad.append(CI_bad)

        recall_mean_good.append(mean_recall_good)
        recall_CI_good.append(CI_good)

        precisions_good, mean_precision_good, CI_good = compute_precision(TI_good, gold_positions_good, summary_analyser)
        precisions_bad, mean_precision_bad, CI_bad = compute_precision(TI_bad, gold_positions_bad, summary_analyser)

        precision_mean_bad.append(mean_precision_bad)
        precision_CI_bad.append(CI_bad)

        precision_mean_good.append(mean_precision_good)
        precision_CI_good.append(CI_good)

        X = [1 for _ in recalls_good] + [0 for _ in recalls_bad]
        Y = recalls_good + recalls_bad
        cor_R = pointbiserialr(x = X, y = Y)[0]
        correlations_recall.append(cor_R)

        X = [1 for _ in precisions_good] + [0 for _ in precisions_bad]
        Y = precisions_good + precisions_bad
        cor_P = pointbiserialr(x = X, y = Y)[0]
        correlations_precision.append(cor_P)

        F_good = [2.0 * (rec * prec) / (rec + prec + 1e-10) for rec,prec in zip(recalls_good, precisions_good)]
        F_bad = [2.0 * (rec * prec) / (rec + prec + 1e-10) for rec,prec in zip(recalls_bad, precisions_bad)]
        X = [1 for _ in F_good] + [0 for _ in F_bad]
        Y = F_good + F_bad
        cor_F = pointbiserialr(x = X, y = Y)[0]
        correlations_F.append(cor_F)

        TI_good = [t[0:w] for t in explained_attention_good]
        TI_bad = [t[0:w] for t in explained_attention_bad]

        attention_good, CI_good = compute_explained_attention(TI_good, summary_analyser)
        attention_bad, CI_bad = compute_explained_attention(TI_bad, summary_analyser)

        explained_attention_mean_bad.append(attention_bad)
        explained_attention_CI_bad.append(CI_bad)

        explained_attention_mean_good.append(attention_good)
        explained_attention_CI_good.append(CI_good)
    
    # https://stackoverflow.com/questions/59747313/how-can-i-plot-a-confidence-interval-in-python
    # https://matplotlib.org/stable/api/markers_api.html
    plt.title("Attention Recall Plot", fontsize = 20)
    plt.xlabel("w", fontsize = 20)
    plt.ylabel("Recall", fontsize = 20)
    plt.plot(w_values, recall_mean_bad, label = "Hard", c = "red", marker = ".")
    plt.fill_between(w_values, [c1 for (c1, c2) in recall_CI_bad], [c2 for (c1, c2) in recall_CI_bad], color = "r", alpha = 0.1)
    plt.plot(w_values, recall_mean_good, label = "Easy", c = "blue", marker = ".")
    plt.fill_between(w_values, [c1 for (c1, c2) in recall_CI_good], [c2 for (c1, c2) in recall_CI_good], color = "b", alpha = 0.1)
    plt.tick_params("x", labelsize = 18)
    plt.tick_params("y", labelsize = 18)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.title("Attention Precision Plot", fontsize = 20)
    plt.xlabel("w", fontsize = 20)
    plt.ylabel("Precision", fontsize = 20)
    plt.plot(w_values, precision_mean_bad, label = "Hard", c = "red" , marker = ".")
    plt.fill_between(w_values, [c1 for (c1, c2) in precision_CI_bad], [c2 for (c1, c2) in precision_CI_bad], color = "r", alpha = 0.1)
    plt.plot(w_values, precision_mean_good, label = "Easy", c = "blue" , marker = ".")
    plt.fill_between(w_values, [c1 for (c1, c2) in precision_CI_good], [c2 for (c1, c2) in precision_CI_good], color = "b", alpha = 0.1)
    plt.tick_params("x", labelsize = 18)
    plt.tick_params("y", labelsize = 18)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.title("Attention Correlation Plot", fontsize = 20)
    plt.xlabel("w", fontsize = 20)
    plt.ylabel("Correlation", fontsize = 20)
    plt.plot(w_values, correlations_F, label = "F score", marker = ".")
    plt.plot(w_values, correlations_recall, label = "Recall", marker = ".")
    plt.plot(w_values, correlations_precision, label = "Precision", marker = ".")
    plt.tick_params("x", labelsize = 18)
    plt.tick_params("y", labelsize = 18)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    entropies_good = [entropy(pk = lst) for lst in explained_attention_good]
    (c1_good, c2_good) = summary_analyser.bootstrap(entropies_good, bootstrap_stats_function = np.mean, alpha = 0.05)

    entropies_bad = [entropy(pk = lst) for lst in explained_attention_bad]
    (c1_bad, c2_bad) = summary_analyser.bootstrap(entropies_bad, bootstrap_stats_function = np.mean, alpha = 0.05)

    res_string = "Entropy_good 95% CI: [{},{}]\nEntropy_Bad 95% CI: [{},{}]".format(round(c1_good, 3), round(c2_good, 3), round(c1_bad, 3), round(c2_bad, 3))

    plt.title("Explained Attention Plot", fontsize = 20)
    plt.xlabel("w", fontsize = 20)
    plt.ylabel("Prop_att", fontsize = 20)
    plt.plot(w_values, explained_attention_mean_bad, label = "Hard", c = "red", marker = ".")
    plt.fill_between(w_values, [c1 for (c1, c2) in explained_attention_CI_bad], [c2 for (c1, c2) in explained_attention_CI_bad], color = "r", alpha = 0.1)
    plt.plot(w_values, explained_attention_mean_good, label = "Easy", c = "blue", marker = ".")
    plt.fill_between(w_values, [c1 for (c1, c2) in explained_attention_CI_good], [c2 for (c1, c2) in explained_attention_CI_good], color = "b", alpha = 0.1)
    plt.tick_params("x", labelsize = 18)
    plt.tick_params("y", labelsize = 18)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(res_string)


    n_examples = 10
    print("\nExamples from easy samples:")
    present_examples(top_tokens_good[0:n_examples], good_queries, good_answers[0:n_examples], answers_good[0:n_examples])
    print("\nExamples from hard samples:")
    present_examples(top_tokens_bad[0:n_examples], bad_queries, bad_answers[0:n_examples], answers_bad[0:n_examples])


def present_examples(top_attentions, queries, true_answers, predicted_answers):
    for attention,query,true_answer,predicted_answer in zip(top_attentions, queries, true_answers, predicted_answers):
        print("\nNEW EXAMPLE")
        print("Query = {}".format(query))
        print("True Answer = {}".format(true_answer))
        print("Predicted Answer = {}".format(predicted_answer))
        print("Top 10 attended input tokens = {}".format(attention[0:20]))

def compute_explained_attention(top_attentions, summary_analyser):
    results = []

    for lst in top_attentions:
        results.append(sum(lst))
    
    CI = summary_analyser.bootstrap(results, bootstrap_stats_function = np.mean, alpha = 0.05)
    return sum(results) / len(results), CI

def compute_recall(top_k_indeces, true_indeces, summary_analyser):
    results = []
    num_empty = 0
    for (top_indeces, answer_indeces) in zip(top_k_indeces, true_indeces):
        
        if len(answer_indeces) == 0:
            num_empty += 1
            continue

        num_overlaps = 0
        for answer_idx in answer_indeces:
            if answer_idx in top_indeces:
                num_overlaps += 1 
        
        overlap_ratio = num_overlaps / len(answer_indeces)
        results.append(overlap_ratio)
    
    #print("Empty True Indexes = {}".format(num_empty))

    CI = summary_analyser.bootstrap(results, bootstrap_stats_function = np.mean, alpha = 0.05)
    return results, sum(results) / len(results), CI

def compute_precision(top_k_indeces, true_indeces, summary_analyser):
    results = []
    num_empty = 0
    for (top_indeces, answer_indeces) in zip(top_k_indeces, true_indeces):
        
        if len(top_indeces) == 0:
            num_empty += 1
            continue

        num_overlaps = 0
        for answer_idx in top_indeces:
            if answer_idx in answer_indeces:
                num_overlaps += 1 
        
        overlap_ratio = num_overlaps / len(top_indeces)
        results.append(overlap_ratio)
    
    #print("Empty True Indexes = {}".format(num_empty))

    CI = summary_analyser.bootstrap(results, bootstrap_stats_function = np.mean, alpha = 0.05)
    return results, sum(results) / len(results), CI

def to_prompts(df, prompter):
    questions = list(df["query"])
    contexts = list(df["text"])
    prompts = prompter.construct_prompts_with_context(questions, contexts)
    return prompts


'''
    cross_attention_maps: List of length N_batch. Each entry in the list, has a tuple for every output token.
    Each output token, has a tuple, with an entry for every decoder layer (there are 24 decoder layers). 
    Each of these tuples has shape:
    [1, 32, 1, input_sequence_length + 1], 
    with 32 denoting the number of attention heads. I guess the EOS token is excluded
    from the input tokens.

    Returns: List, with an entry for every sample. Each entry, contains the top 20 cross attended tokens

'''
# https://discuss.huggingface.co/t/google-t5-cross-attentions-output/104541
# https://huggingface.co/google/flan-t5-large/blob/main/config.json
# https://huggingface.co/google/flan-t5-xl/blob/main/config.json
# https://discuss.huggingface.co/t/t5-cross-attention-inconsistent-results/5520
# https://github.com/huggingface/transformers/tree/main/src/transformers/models/t5
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py
def attention_analysis(cross_attention_maps, inputs_tokens):
    N_batch = len(cross_attention_maps)
    token_results = []
    index_results = []
    attention_results = []
    for i in tqdm(range(N_batch)):
        input_tokens = np.array(inputs_tokens[i] + ["EOS"])
        num_input_tokens = len(input_tokens)
    
        attention_map = cross_attention_maps[i]
        output_length = len(attention_map)
        #print("Number of Output Tokens = {}".format(output_length))

        aggregated_attentions = np.zeros((num_input_tokens))

        for j in range(output_length):

            cross_attention_heads = attention_map[j]
            num_decoder_layers = len(cross_attention_heads)

            for k in range(num_decoder_layers):
                cross_attentions = cross_attention_heads[k].cpu().detach().numpy()
                # Index to shape [num_heads, input_sequence_length]
                cross_attentions = cross_attentions[0, :, 0, :]
                # Summed attention values for every input token across all the attention heads
                cross_attentions_agg = np.sum(cross_attentions, axis = 0)
                aggregated_attentions += cross_attentions_agg
        
        attentions_sorted = np.sort(aggregated_attentions)[::-1] / np.sum(aggregated_attentions)
        indeces_sorted = np.argsort(aggregated_attentions)[::-1]
        # To map it to input tokens
        input_tokens_sorted = input_tokens[indeces_sorted]

        token_results.append(input_tokens_sorted)
        index_results.append(indeces_sorted)
        attention_results.append(attentions_sorted)
    
    return token_results, index_results, attention_results
                
                
def print_df(df):
    print("\n")
    print(df[["True Answer", "Predicted Answer"]])
    print("\n")


if __name__ == "__main__":
    main()