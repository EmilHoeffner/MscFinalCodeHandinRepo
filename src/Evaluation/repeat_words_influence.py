import nltk
from nltk.tokenize import word_tokenize

import sys 

sys.path.append("..")

from Filehandling.FileHandler import FileHandler
from Utils.listutils import list_distinct

def repeat_invariance_score(answers):
    repeat_invariant = []
    for ans in answers:
        answer_tokens = word_tokenize(str(ans))
        if len(answer_tokens) == len(list_distinct(answer_tokens)):
            repeat_invariant.append(1)
        else:
            repeat_invariant.append(0)
    
    return sum(repeat_invariant) / len(repeat_invariant)

def main():
    num_args = len(sys.argv)

    if num_args != 2:
        raise Exception("Program must be run with exactly one argument")
    
    data_set_iden = sys.argv[1]
    nltk.download('punkt')

    file_handler = FileHandler()

    df_RAG = file_handler.load_df(dir = "EvaluationFiles", dataset_name = data_set_iden, filename = "M_RAG.csv")
    df_Question = file_handler.load_df(dir = "EvaluationFiles", dataset_name = data_set_iden, filename = "M_question.csv")

    true_answers = list(df_RAG["True Answer"])
    RAG_answers = list(df_RAG["Predicted Answer"])
    question_answers = list(df_Question["Predicted Answer"])

    true_answer_score = repeat_invariance_score(true_answers)
    RAG_answers_score = repeat_invariance_score(RAG_answers)
    question_answers_score = repeat_invariance_score(question_answers)

    print("Ground Truth Answer Score = {}".format(true_answer_score))
    print("RAG Answer Score = {}".format(RAG_answers_score))
    print("Question Answer Score = {}".format(question_answers_score))



if __name__ == "__main__":
    main()