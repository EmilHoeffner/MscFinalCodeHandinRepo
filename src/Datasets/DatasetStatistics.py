import sys 
import os 
import time


from DatasetExplorer import DatasetExplorer

sys.path.append("..")

from Dataloading.Dataloader import DataLoader

# Computation time for Squad_Tune, with 11.000 samples took 1.2 minutes

def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # https://www.nielit.gov.in/gorakhpur/sites/default/files/Gorakhpur/ALevel_1_Python_20May.pdf
    num_args = len(sys.argv)

    if num_args != 2:
        raise Exception("Program must be run with exactly one argument, denoting which dataset to derive statistics over")
    
    data_set_iden = sys.argv[1]

    time_start = time.time()

    print("Loading Data \n")
    dl = DataLoader()
    dataset = dl.load_data(data_set_iden)
    explorer = DatasetExplorer(data_set_iden, dataset)

    print("Computing Token Length Distributions \n")
    explorer.store_context_token_length_distribution()
    explorer.store_question_token_length_distribution()
    explorer.store_answer_token_length_distribution()



    print("Computing Lix Distributions \n")
    explorer.store_context_lix_distribution()
    explorer.store_question_lix_distribution()

    '''
    print("Computing Word Clouds \n")
    explorer.store_context_word_cloud()
    explorer.store_question_word_cloud()
    '''
    
    print("Computing Questions per context \n")
    explorer.store_questions_per_context()

    time_end = time.time()

    time_total = (time_end - time_start) / 60.0

    print("\nComputing Dataset Statistics took {} minutes".format(time_total))


if __name__ == "__main__":
    main()