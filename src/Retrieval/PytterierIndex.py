# https://pyterrier.readthedocs.io/en/latest/terrier/how-to.html#7aeb11d60b004e9f8f63f82a8336a3c8

'''
    Used code from SE 2025 - Lab1.ipynb from the Search Engines course 2024/2025
'''

import pyterrier as pt
import sys
import glob
import os

sys.path.append("..")

from Filehandling.FileHandler import FileHandler


class PytterierIndex:
    def __init__(self, dataset_name):
        self.save_path = FileHandler().storage_path() + f"/IndexFiles/{dataset_name}"

        if not pt.started():
            pt.init()

    '''
        Data: A list with entries {"docno" : id, "context" : context}

        Returns: A reference to the index.
    '''
    # https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder
    def store_index(self, data):
        # Delete prior saved index, since the indexer for some reason cannot overwrite properly.
        files = glob.glob("{}/*".format(self.save_path))

        for f in files:
            os.remove(f)

        indexer = pt.IterDictIndexer(self.save_path, overwrite=True, stopwords = 'terrier', stemmer = None)
        index_ref = indexer.index(data)
        index = pt.IndexFactory.of(index_ref)

        print("Stored index with the following statistics:\n")
        print(index.getCollectionStatistics().toString())

        return index 
    
    '''
        Loads the index stored by the store_index function
    '''
    def load_index(self):
        load_path = self.save_path + "/data.properties"
        index_ref = pt.IndexRef.of(load_path)
        index = pt.IndexFactory.of(index_ref)

        return index