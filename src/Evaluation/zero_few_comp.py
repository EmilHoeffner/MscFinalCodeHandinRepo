
import os
import sys 
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score 
from sklearn.metrics import f1_score

sys.path.append("..")

from Filehandling.FileHandler import FileHandler

from Utils import plotUtils


def switch_statistics(zero_scores, few_scores, stats_name):
    agree_1 = 0
    agree_0 = 0
    zero_1_few_0 = 0
    zero_0_few_1 = 0

    for (zero, few) in zip(zero_scores, few_scores):
        if zero == 1 and few == 1:
            agree_1 += 1
        elif zero == 0 and few == 0:
            agree_0 += 1 
        elif zero == 1 and few == 0:
            zero_1_few_0 += 1
        elif zero == 0 and few == 1:
            zero_0_few_1 += 1 
        else:
            raise Exception("Invalid Branch reached")
    
    print(f"\nSwitch Statistics for {stats_name}:\n")

    print(f"Zero == 1 and Few == 1 => {agree_1}")
    print(f"Zero == 0 and Few == 0 => {agree_0}")
    print(f"Zero == 1 and Few == 0 => {zero_1_few_0}")
    print(f"Zero == 0 and Few == 1 => {zero_0_few_1}")
    



def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # https://www.nielit.gov.in/gorakhpur/sites/default/files/Gorakhpur/ALevel_1_Python_20May.pdf
    num_args = len(sys.argv)

    if num_args != 2:
        raise Exception("Program must be run with exactly two argument, denoting which dataset to evaluate over")
    
    data_set_iden = sys.argv[1]

    file_handler = FileHandler()

    zero = file_handler.load_df(dir = "EvaluationFiles", dataset_name = data_set_iden, filename = "M_RAG.csv")
    few = file_handler.load_df(dir = "EvaluationFiles", dataset_name = data_set_iden + "_Few", filename = "M_RAG.csv")

    zero_vals = list(zero["F1"])
    few_vals = list(few["F1"])

    assert(len(zero_vals) == len(few_vals))

    aggrement_score = 0 

    for s_zero, s_few in zip(zero_vals, few_vals):
        if abs(s_zero - s_few) <= 0.01:
            aggrement_score += 1
    
    aggrement_score /= len(zero_vals)

    cor = pearsonr(zero_vals, few_vals, alternative = "two-sided")
    val = cor[0]
    p_val = cor[1]

    value_str = f"\ncor = {round(val,5)} at p = {round(p_val,5)}\naggrement = {round(aggrement_score, 5)}"

    print("Value Str = {}".format(value_str))

    title = f"{data_set_iden}: Zero Shot vs Few Shot"
    x_label = "F1: Zero Shot"
    y_label = "F1: Few Shot"
    save_path = f"{file_handler.storage_path()}/TuningPlots/scatterHist_{data_set_iden}.png"

    #plotUtils.scatter_hist(zero_vals, few_vals, title, x_label, y_label, save_path)
    plotUtils.scatter_density_2d(zero_vals, few_vals, title, x_label, y_label, save_path)

    
    print("Statistics over F1 = 0 scores:\n")

    zero_zero_scores = [1 if zero_val == 0.0 else 0 for zero_val in zero_vals]
    few_zero_scores = [1 if few_val == 0.0 else 0 for few_val in few_vals]

    print(f"Frequency of zero 0 scores = {sum(zero_zero_scores)}")
    print(f"Frequency of few 0 scores = {sum(few_zero_scores)}")

    accuracy = accuracy_score(few_zero_scores, zero_zero_scores, )
    precision = precision_score(few_zero_scores, zero_zero_scores)
    recall = recall_score(few_zero_scores, zero_zero_scores)
    F1 = f1_score(few_zero_scores, zero_zero_scores)

    print("Accuracy = {}".format(accuracy))
    print("Precision = {}".format(precision))
    print("Recall = {}".format(recall))
    print("F1 = {}".format(F1))


    print("\nStatistics over 0 < F1 < 1 scores\n")
    zero_middle_scores = [1 if zero_val > 0.0 and zero_val < 1.0 else 0 for zero_val in zero_vals]
    few_middle_scores = [1 if few_val > 0.0 and few_val < 1.0 else 0 for few_val in few_vals]

    print(f"Frequency of Zero Middle scores = {sum(zero_middle_scores)}")
    print(f"Frequency of Few Middle scores = {sum(few_middle_scores)}")

    accuracy = accuracy_score(few_middle_scores, zero_middle_scores)
    precision = precision_score(few_middle_scores, zero_middle_scores)
    recall = recall_score(few_middle_scores, zero_middle_scores)
    F1 = f1_score(few_middle_scores, zero_middle_scores)

    print("Accuracy = {}".format(accuracy))
    print("Precision = {}".format(precision))
    print("Recall = {}".format(recall))
    print("F1 = {}".format(F1))

    print("\nStatistics over F1 = 1 scores:\n")

    zero_perfect_scores = [1 if zero_val == 1.0 else 0 for zero_val in zero_vals]
    few_perfect_scores = [1 if few_val == 1.0 else 0 for few_val in few_vals]

    print(f"Frequency of zero perfect scores = {sum(zero_perfect_scores)}")
    print(f"Frequency of few perfect scores = {sum(few_perfect_scores)}")

    accuracy = accuracy_score(few_perfect_scores, zero_perfect_scores)
    precision = precision_score(few_perfect_scores, zero_perfect_scores)
    recall = recall_score(few_perfect_scores, zero_perfect_scores)
    F1 = f1_score(few_perfect_scores, zero_perfect_scores)

    print("Accuracy = {}".format(accuracy))
    print("Precision = {}".format(precision))
    print("Recall = {}".format(recall))
    print("F1 = {}".format(F1))


    print("\nIDK Statistics\n")

    idk_vals_few = np.array(list(few["IDK"]))
    idk_vals_zero = np.array(list(zero["IDK"]))

    print("Frequency Zero IDK = 1 scores = {}".format(np.sum(idk_vals_zero)))
    print("Frequency Few IDK = 1 scores = {}".format(np.sum(idk_vals_few)))

    accuracy = accuracy_score(idk_vals_few, idk_vals_zero)
    precision = precision_score(idk_vals_few, idk_vals_zero)
    recall = recall_score(idk_vals_few, idk_vals_zero)
    F1 = f1_score(idk_vals_few, idk_vals_zero)

    F1_switch_scores = []

    for i,(s1, s2) in enumerate(zip(idk_vals_zero, idk_vals_few)):
        if s1 == 1 and s2 == 0:
            F1_switch_scores.append(few_vals[i])

    print("Accuracy = {}".format(accuracy))
    print("Precision = {}".format(precision))
    print("Recall = {}".format(recall))
    print("F1 = {}".format(F1))
    print("F1 Switch Mean Score = {}".format(sum(F1_switch_scores) / len(F1_switch_scores)))

    print("\n")

    switch_statistics(idk_vals_zero, idk_vals_few, "IDK")



if __name__ == "__main__":
    main()