
# https://en.wikipedia.org/wiki/Grammy_Award_for_Album_of_the_Year

# Does not seem like Beyonce has ever lost to Drake

from Evaluater import Evaluater

import sys

import time

# Time on Squad

# M_Question: 3.661060118675232 min

# M_RAG:  9.205534180005392 min

# M_context: 8.655711774031321 min


sys.path.append("..")

from Filehandling.FileHandler import FileHandler

def main():
    num_args = len(sys.argv)

    if num_args != 2:
        raise Exception("Program must be run with exactly one argument, denoting which dataset to evaluate over")
    
    data_set_iden = sys.argv[1]

    evaluater = Evaluater()
    file_handler = FileHandler()
    
    df_question = file_handler.load_df(dir = "Collab", dataset_name = data_set_iden, filename = "M_question.csv")
    df_RAG = file_handler.load_df(dir = "Collab", dataset_name = data_set_iden, filename = "M_RAG.csv")
    df_context = file_handler.load_df(dir = "Collab", dataset_name = data_set_iden, filename = "M_context.csv")

    for df,desc in [(df_question, "question"), (df_RAG, "RAG"), (df_context, "context")]:
        time_start = time.time()
        qids = list(df["qid"])
        questions = list(df["Question"])
        true_answers = list(df["True Answer"])
        predicted_answers = list(df["Predicted Answer"])

        if desc == "question":
            df = evaluater.hallucination_detection_without_context(qids = qids, questions = questions, predicted_answers = predicted_answers, true_answers = true_answers)
        else:
            contexts = list(df["context"])
            df = evaluater.hallucination_detection_with_context(qids = qids, questions = questions, contexts = contexts, predicted_answers = predicted_answers, true_answers = true_answers)
        
        file_handler.save_df(df = df, dir = "EvaluationFiles", dataset_name = data_set_iden, filename = f"M_{desc}_Hallucination.csv", header = True)
        time_end = time.time()
        time_total = (time_end - time_start) / 60.0
        print("Done with Hallucination for M_{} in {} minutes".format(desc, time_total))


if __name__ == "__main__":
    main()