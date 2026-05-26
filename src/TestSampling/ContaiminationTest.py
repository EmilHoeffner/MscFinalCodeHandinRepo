

import sys

sys.path.append("..")

from Filehandling.FileHandler import FileHandler


def main():
    num_args = len(sys.argv)

    if num_args != 2:
        raise Exception("Program must be run with exactly one argument")
    
    data_set_iden = sys.argv[1]

    file_handler = FileHandler()

    df_queries_tune = list(file_handler.load_df(dir = "DataFiles", dataset_name = f"{data_set_iden}_Tune", filename = "queries.csv")["query"])
    df_queries_test = list(file_handler.load_df(dir = "DataFiles", dataset_name = f"{data_set_iden}_Test", filename = "queries.csv")["query"])

    df_queries_tune = [str(q) for q in df_queries_tune]
    df_queries_test = [str(q) for q in df_queries_test]

    df_docs_tune = list(file_handler.load_df(dir = "DataFiles", dataset_name = f"{data_set_iden}_Tune", filename = "docs.csv")["text"])
    df_docs_test = list(file_handler.load_df(dir = "DataFiles", dataset_name = f"{data_set_iden}_Test", filename = "docs.csv")["text"])

    df_docs_tune = [str(q) for q in df_docs_tune]
    df_docs_test = [str(q) for q in df_docs_test]



    for q in df_queries_tune:
        assert(q not in df_queries_test)

    for d in df_docs_tune:
        assert(d not in df_docs_test)

    







if __name__ == "__main__":
    main()