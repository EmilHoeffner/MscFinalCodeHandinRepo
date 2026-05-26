
from datasets import load_dataset

# https://huggingface.co/datasets/google/boolq?library=datasets

class BoolQ:
    def __init__(self):
        ds = load_dataset("google/boolq")
        ds = ds["train"]
        print(ds[0])
