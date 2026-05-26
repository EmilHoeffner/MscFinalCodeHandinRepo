# https://www.geeksforgeeks.org/python/get-parent-of-current-directory-using-python/

import os
import pandas as pd

class FileHandler:

    def __init__(self, iden):
        cwd = os.getcwd()
        # Set the path to the Storage folder
        self.path = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir)) + f"/{iden}"

        print(self.path)

    def storage_path(self):
        return self.path

    '''
        df: A pandas dataframe
        dir: The directory in Storage to save the file in (a string)
        filename: The filename
        header: A boolean, that denotes whether to include the header in the saved csv.
    '''
    # https://stackoverflow.com/questions/41087619/pandas-merge-how-to-avoid-unnamed-column
    def save_df(self, df, dir, dataset_name, filename, header, index = False):
        path = self.path + f"/{dir}/{dataset_name}/{filename}"
        df.to_csv(path, header = header, index = index)

    '''
        dir: The directory in Storage to load the dataframe
        filename: The filename
    '''
    def load_df(self, dir, dataset_name, filename):
        return pd.read_csv(self.path + f"/{dir}/{dataset_name}/{filename}")
    
    # https://stackoverflow.com/questions/82831/how-do-i-check-whether-a-file-exists-without-exceptions
    def file_exists(self, dir, dataset_name, filename):
        path = self.path + f"/{dir}/{dataset_name}/{filename}"
        return os.path.exists(path)
