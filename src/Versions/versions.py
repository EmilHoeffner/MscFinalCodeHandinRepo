import numpy
import pandas 
import matplotlib 
import nltk 
import scipy
import sklearn
import statsmodels
import transformers
import datasets
import torch 
import torchmetrics
import sentence_transformers
import pyterrier
import readcalc
import huggingface_hub


def main():
    print("Numpy Version = {}".format(numpy.__version__))
    print("Pandas Version = {}".format(pandas.__version__))
    print("Matplotlib Version = {}".format(matplotlib.__version__))
    print("NLTK Version = {}".format(nltk.__version__))
    print("Scipy Version = {}".format(scipy.__version__))
    print("Sklearn Version = {}".format(sklearn.__version__))
    print("Statsmodels Version = {}".format(statsmodels.__version__))
    print("Transformers Version = {}".format(transformers.__version__))
    print("Datasets Version = {}".format(datasets.__version__))
    print("Torch Version = {}".format(torch.__version__))
    print("TorchMetrics Version = {}".format(torchmetrics.__version__))
    print("Sentence Transformers Version = {}".format(sentence_transformers.__version__))
    print("Pyterrier Version = {}".format(pyterrier.__version__))
    #print("Readcalc Version = {}".format(readcalc.__version__))
    print("Huggingface Hub Version = {}".format(huggingface_hub.__version__))


if __name__ == "__main__":
    main()