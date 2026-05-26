
import sys

sys.path.append("..")

from Filehandling.FileHandler import FileHandler
from Utils.listutils import list_distinct
import nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# https://www.geeksforgeeks.org/nlp/removing-stop-words-nltk-python/

def main():
    num_args = len(sys.argv)

    if num_args != 2:
        raise Exception("Program must be run with exactly one argument")
    
    data_set_iden = sys.argv[1]

    stop_words = set(stopwords.words('english'))

    file_handler = FileHandler()

    queries = file_handler.load_df(dir = "DataFiles", dataset_name = data_set_iden, filename = "queries.csv")

    queries_text = list(queries["query"])

    cnt = 0

    for q in queries_text:
        tokens = word_tokenize(q)
        filtered_tokens = [word for word in tokens if word not in stop_words]
        if filtered_tokens == list_distinct(filtered_tokens):
            cnt += 1
    
    ratio = cnt / len(queries_text)

    print(ratio)


if __name__ == "__main__":
    main()