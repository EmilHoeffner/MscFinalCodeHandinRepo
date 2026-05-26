from Evaluater import Evaluater

import sys

import time

sys.path.append("..")

from Filehandling.FileHandler import FileHandler

def main():
    num_args = len(sys.argv)

    if num_args != 2:
        raise Exception("Program must be run with exactly one argument, denoting which dataset to evaluate over")
    
    data_set_iden = sys.argv[1]

    evaluater = Evaluater()
    file_handler = FileHandler()
    
    df_question = file_handler.load_df(dir = "EvaluationFiles", dataset_name = data_set_iden, filename = "M_question.csv")
    df_RAG = file_handler.load_df(dir = "EvaluationFiles", dataset_name = data_set_iden, filename = "M_RAG.csv")
    df_context = file_handler.load_df(dir = "EvaluationFiles", dataset_name = data_set_iden, filename = "M_context.csv")

    df_queries = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "queries.csv")

    df_question = df_question.merge(df_queries, on="qid", how = "left", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
    df_RAG = df_RAG.merge(df_queries, on="qid", how = "left", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
    df_context = df_context.merge(df_queries, on="qid", how = "left", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

    for df,desc in [(df_question, "question"), (df_RAG, "RAG"), (df_context, "context")]:
        time_start = time.time()
        qids = list(df["qid"])
        questions = list(df["query"])
        true_answers = list(df["True Answer"])
        predicted_answers = list(df["Predicted Answer"])

        print(f"Evaluating M_{desc}")
        df = evaluater.LLM_As_Judge(qids = qids, questions = questions, predicted_answers = predicted_answers, true_answers = true_answers)
        
        file_handler.save_df(df = df, dir = "EvaluationFiles", dataset_name = data_set_iden, filename = f"M_{desc}_LLM_As_Judge.csv", header = True)
        time_end = time.time()
        time_total = (time_end - time_start) / 60.0
        print("Done with LLM As Judge for M_{} in {} minutes".format(desc, time_total))


if __name__ == "__main__":
    main()