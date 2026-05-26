import pyterrier as pt

'''
    Used code from SE 2025 - Lab2.ipynb from the Search Engines course 2024/2025
'''

# http://terrier.org/docs/current/javadoc/org/terrier/matching/models/package-summary.html (overview of the retrieval models)

class BM25:

    # Default is the default parameters in pytterier, with the exception of k3 = 0
    def __init__(self, index, b = 0.75, k1 = 1.2, k3 = 0, num_results = 10):
        if not pt.started():
            pt.init()
        self.model = pt.terrier.Retriever(index, num_results = num_results, controls={'wmodel': 'BM25', 'bm25.k_1': k1, 'bm25.b': b, 'bm25.k_3': k3})

    '''
        index: A pointer to the index.
        df: pandas dataframe with entries qid and text. 

        Returns: Dataframe with 
    '''
    # https://www.geeksforgeeks.org/pandas/pandas-select-columns/
    def retrieve(self, df):
        df = self.model(df)
        selected_columns = df.loc[:, ['qid', 'docno', 'score', 'rank']]
        return selected_columns
