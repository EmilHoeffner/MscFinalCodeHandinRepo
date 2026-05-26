
from dataloader import DataLoader
import os 
import sys 

import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

sys.path.append("..")

from TextAnalysis.TFIDFModel import TFIDFModel
from LLM.SentenceBERT import SentenceBERT
from Filehandling.FileHandler import FileHandler



# https://stackoverflow.com/questions/12487060/color-according-to-class-labels
def visualisation_plot_summary_statistics(features, y, y_name, dir_path, method):
    good_indices = []
    bad_indices = []

    lower_bound = 0.5 if y_name == "BERT" else 0.0

    N = len(y)

    X = np.zeros((N, len(features)))

    for i,lst in enumerate(features):
        X[:, i] = np.array(lst)

    for i, y_value in enumerate(y):
        if y_value <= lower_bound:
            bad_indices.append(i)
        else:
            good_indices.append(i)

    N_good = len(good_indices)

    if method == "PCA":
        model = PCA(n_components = 2)
    elif method == "TSNE":
        model  = TSNE(n_components = 2)
    
    print("Starting Dimensionality Reduction Transformation")
    X_transformed = model.fit_transform(X[(good_indices + bad_indices), :])

    assert(len(X_transformed) == N_good + len(bad_indices))

    title = f"TFIDF with {method} with y = {y_name}"

    if method == "PCA":
        explained_variance = np.sum(model.explained_variance_ratio_)
        title += " EV = {:.4f}".format(round(explained_variance, 4))


    plt.title(title, fontsize = 16)
    plt.scatter(x = X_transformed[0:N_good, 0], y = X_transformed[0:N_good, 1], c = "blue", label = "F1 = 1")
    plt.scatter(x = X_transformed[N_good:, 0], y = X_transformed[N_good:, 1], c = "red", label = "F1 = 0")
    plt.legend()
    plt.savefig(dir_path + "Summary_{}_{}".format(method, y_name))
    plt.close()

def visualisation_plot_Embeddings(X, y, y_name, dir_path, reduction_method, embed_method):
    good_indices = []
    bad_indices = []

    lower_bound = 0.5 if y_name == "BERT" else 0.0

    for i, y_value in enumerate(y):
        if y_value <= lower_bound:
            bad_indices.append(i)
        else:
            good_indices.append(i)

    N_good = len(good_indices)

    if reduction_method == "PCA":
        model = PCA(n_components = 2)
    elif reduction_method == "TSNE":
        model  = TSNE(n_components = 2)
    
    print("Starting Dimensionality Reduction Transformation")
    X_transformed = model.fit_transform(X[(good_indices + bad_indices), :])

    assert(len(X_transformed) == N_good + len(bad_indices))

    title = f"{embed_method} with {reduction_method} with y = {y_name}"

    if reduction_method == "PCA":
        explained_variance = np.sum(model.explained_variance_ratio_)
        title += " EV = {:.4f}".format(round(explained_variance, 4))


    plt.title(title, fontsize = 16)
    plt.scatter(x = X_transformed[0:N_good, 0], y = X_transformed[0:N_good, 1], c = "blue", label = "F1 = 1")
    plt.scatter(x = X_transformed[N_good:, 0], y = X_transformed[N_good:, 1], c = "red", label = "F1 = 0")
    plt.legend()
    plt.savefig(dir_path + "{}_{}_{}".format(embed_method, reduction_method, y_name))
    plt.close()



def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # https://www.nielit.gov.in/gorakhpur/sites/default/files/Gorakhpur/ALevel_1_Python_20May.pdf
    num_args = len(sys.argv)

    if num_args != 3:
        raise Exception("Program must be run with exactly two arguments")
    
    data_set_iden = sys.argv[1]
    y_name = sys.argv[2]

    dl = DataLoader(data_set_iden)
    file_handler = FileHandler()

    features, feature_names = dl.named_train_features()
    y_train = dl.load_train_feature(y_name)

    questions_train = dl.load_train_feature("question")
    contexts_train = dl.load_train_feature("context")

    dir_path = file_handler.storage_path() + f"/CorrelationAnalysis/{data_set_iden}/Visualisation/"

    # Summary Statistics
    visualisation_plot_summary_statistics(features, y_train, y_name, dir_path, method = "TSNE")
    visualisation_plot_summary_statistics(features, y_train, y_name, dir_path, method = "PCA")

    # TFIDF:
    tfidf_model = TFIDFModel()
    X_tfidf_train = tfidf_model.fit(questions_train, contexts_train)
    visualisation_plot_Embeddings(X_tfidf_train, y_train, y_name, dir_path, "TSNE", embed_method = "TFIDF")
    visualisation_plot_Embeddings(X_tfidf_train, y_train, y_name, dir_path, "PCA", embed_method = "TFIDF")

    # Sentence BERT:
    sentence_BERT = SentenceBERT()
    X_BERT_question_train = sentence_BERT.embed(questions_train)
    X_BERT_context_train = sentence_BERT.embed(contexts_train)
    X_BERT_total_train = np.concatenate((X_BERT_question_train, X_BERT_context_train), axis = 1)

    visualisation_plot_Embeddings(X_BERT_total_train, y_train, y_name, dir_path, "TSNE", embed_method = "SentenceBERT")
    visualisation_plot_Embeddings(X_BERT_total_train, y_train, y_name, dir_path, "PCA", embed_method = "SentenceBERT")



if __name__ == "__main__":
    main()