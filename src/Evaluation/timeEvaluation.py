
import sys 

sys.path.append("..")

from Filehandling.FileHandler import FileHandler

def main():
    num_args = len(sys.argv)

    if num_args != 2:
        raise Exception("Program must be run with exactly one argument, denoting which dataset to evaluate over")
    
    data_set_iden = sys.argv[1]

    file_handler = FileHandler()

    M_question_time = sum(list(file_handler.load_df(dir = "EvaluationFiles", dataset_name = data_set_iden,
                                           filename = "M_question.csv")["Time"]))
    M_RAG_time = sum(list(file_handler.load_df(dir = "EvaluationFiles", dataset_name = data_set_iden,
                                           filename = "M_RAG.csv")["Time"]))
    M_ceil_time = sum(list(file_handler.load_df(dir = "EvaluationFiles", dataset_name = data_set_iden,
                                           filename = "M_context.csv")["Time"]))
    
    print("M_question time = {}".format(M_question_time))
    print("M_RAG time = {}".format(M_RAG_time))
    print("M_ceil time = {}".format(M_ceil_time))




if __name__ == "__main__":
    main()