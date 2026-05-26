import sys

sys.path.append("..")

from Filehandling.FileHandler import FileHandler
from Utils.listutils import list_distinct


def main():
    num_args = len(sys.argv)

    if num_args != 2:
        raise Exception("Program must be run with exactly one argument")
    
    data_set_iden = sys.argv[1]

    file_handler = FileHandler()

    df = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "docs.csv")
    docs = list(df["text"]) 

    assert(docs == list_distinct(docs))


if __name__ == "__main__":
    main()