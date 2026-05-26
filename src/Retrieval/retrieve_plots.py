import sys 
import matplotlib.pyplot as plt

sys.path.append("..")

from Filehandling.FileHandler import FileHandler

'''
    An example of name is BM25(b = 0.1,k1 = 1.0)
'''
def b(name):
    s = name.split(",")
    b_value = float(s[0].split("=")[1])

    return b_value

'''
    An example of name is BM25(b = 0.1,k1 = 1.0)
'''
def k1(name):
    s = name.split(",")
    k1_value = float(s[1].split("=")[1].replace(")", ""))

    return k1_value


def retrieve_plot(x_squad, x_wiki, y_squad, y_wiki, title, x_label, y_label, save_path, include_legend = False):
    #plt.title(title, fontsize = 18)
    plt.scatter(x_squad, y_squad, color = "blue", label = "Squad")
    plt.scatter(x_wiki, y_wiki, color = "red", label = "Wiki")
    plt.xlabel(x_label, fontsize = 18)
    plt.ylabel(y_label, fontsize = 18)
    plt.tick_params("x", labelsize = 18)
    plt.tick_params("y", labelsize = 18)

    if include_legend:
        plt.legend(fontsize = 18)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



def main():    
    file_handler = FileHandler()

    storage_path = file_handler.storage_path()

    results_squad = file_handler.load_df(dir = "RetrievalFiles", dataset_name = "Squad_Tune", filename = "EvaluationFile/Eval.csv")
    results_wiki = file_handler.load_df(dir = "RetrievalFiles", dataset_name = "WikiMultiHop_Tune", filename = "EvaluationFile/Eval.csv")

    names_squad = list(results_squad["name"])
    rec1_squad = list(results_squad["recall_1"])
    rec5_squad = list(results_squad["recall_5"])
    rec10_squad = list(results_squad["recall_10"])
    MRR_squad = list(results_squad["recip_rank"])
    bs_squad = [b(name) for name in names_squad]
    k1s_squad = [k1(name) for name in names_squad]

    names_wiki = list(results_wiki["name"])
    rec1_wiki = list(results_wiki["recall_1"])
    rec5_wiki = list(results_wiki["recall_5"])
    rec10_wiki = list(results_wiki["recall_10"])
    MRR_wiki = list(results_wiki["recip_rank"])
    bs_wiki = [b(name) for name in names_wiki]
    k1s_wiki = [k1(name) for name in names_wiki]

    
    B_squad = [b for b in bs_squad]
    K1_squad = [k1 for k1 in k1s_squad]

    B_wiki = [b for b in bs_wiki]
    K1_wiki = [k1 for k1 in k1s_wiki]


    # MRR plots:

    retrieve_plot(x_squad = B_squad, x_wiki = B_wiki, y_squad = MRR_squad, y_wiki = MRR_wiki,
                  title = "MRR dependance on b", x_label = "b", y_label = "MRR",
                  save_path = f"{storage_path}/TuningPlots/bvsMRR.png", include_legend = True)

    retrieve_plot(x_squad = K1_squad, x_wiki = K1_wiki, y_squad = MRR_squad, y_wiki = MRR_wiki,
                  title = "MRR dependance on k1", x_label = "k1", y_label = "MRR",
                  save_path = f"{storage_path}/TuningPlots/k1vsMRR.png")
    

    # Recall Plots
    
    retrieve_plot(x_squad = B_squad, x_wiki = B_wiki, y_squad = rec1_squad, y_wiki = rec1_wiki,
                  title = "Recall@1 dependance on b", x_label = "b", y_label = "Recall@1",
                  save_path = f"{storage_path}/TuningPlots/bvsrec1.png")

    retrieve_plot(x_squad = K1_squad, x_wiki = K1_wiki, y_squad = rec1_squad, y_wiki = rec1_wiki,
                  title = "Recall@1 dependance on k1", x_label = "k1", y_label = "Recall@1",
                  save_path = f"{storage_path}/TuningPlots/k1vsrec1.png")
    

    retrieve_plot(x_squad = B_squad, x_wiki = B_wiki, y_squad = rec5_squad, y_wiki = rec5_wiki,
                  title = "Recall@5 dependance on b", x_label = "b", y_label = "Recall@5",
                  save_path = f"{storage_path}/TuningPlots/bvsrec5.png")

    retrieve_plot(x_squad = K1_squad, x_wiki = K1_wiki, y_squad = rec5_squad, y_wiki = rec5_wiki,
                  title = "Recall@5 dependance on k1", x_label = "k1", y_label = "Recall@5",
                  save_path = f"{storage_path}/TuningPlots/k1vsrec5.png")

    



if __name__ == "__main__":
    main()
