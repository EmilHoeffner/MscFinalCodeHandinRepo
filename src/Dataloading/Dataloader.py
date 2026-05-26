import sys 


sys.path.append("..")

from Datasets.Squad import Squad 
from Datasets.WikiMultiHop import WikiMultiHop

class DataLoader:

    def __init__(self):
        pass 

    def load_data(self, data_set_iden):

        if data_set_iden == "Squad_Dummy":
            dataset = Squad("Dummy")
        elif data_set_iden == "Squad_Tune":
            dataset = Squad("Tune")
        elif data_set_iden == "Squad_Test":
            dataset = Squad("Test")
        elif data_set_iden == "Squad_TrainFull":
            dataset = Squad("TrainFull")
        elif data_set_iden == "Squad_TestFull":
            dataset = Squad("TestFull")
        elif data_set_iden == "WikiMultiHop_Dummy":
            dataset = WikiMultiHop("Dummy")
        elif data_set_iden == "WikiMultiHop_Tune":
            dataset = WikiMultiHop("Tune")
        elif data_set_iden == "WikiMultiHop_Test":
            dataset = WikiMultiHop("Test")
        elif data_set_iden == "WikiMultiHop_ValidationFull":
            dataset = WikiMultiHop("ValidationFull")
        else:
            raise Exception("Unknown dataset identifier")
        
        return dataset