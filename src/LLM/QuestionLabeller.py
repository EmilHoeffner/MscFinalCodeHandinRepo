
from Flan import FlanT5
import sys 
import pandas as pd
from tqdm import tqdm

#from setfit import SetFitModel, Trainer, TrainingArguments
from Flan import FlanT5

sys.path.append("..")

from Filehandling.FileHandler import FileHandler

# https://huggingface.co/docs/setfit/how_to/zero_shot
# https://huggingface.co/blog/setfit
# https://discuss.huggingface.co/t/what-is-the-best-way-to-classify-my-content-into-tags/145561/2

# NOTE: Did not work well using Flan T5 large.

class QuestionLabeller:
    
    def __init__(self):
        self.labeller = FlanT5()

    '''
        questions: List of strings
    '''
    def label(self, questions):
        prompts = [self.question_to_prompt(question) for question in questions]
        return [self.labeller.answer([prompt])[0][0] for prompt in tqdm(prompts)]

    
    def question_to_prompt(self, question):
        prompt = f"""
        Label the question <Q5> as Definite or Indefinite. A question is indefinite, if it is not clear 
        who or what the subject in the sentence refers to. A question is definite, if it is clear exactly what or who
        the subject of the sentence refers to.

        Some Examples:
        <Q1>: Who is the president of united states?
        <A1>: Definite
        <Q2>: what does he mean?
        <A2>: Indefinite
        <Q3>: What did Federer think about his match versus nadal at wimbledon 2008?
        <A3>: Definite
        <Q4>: which of the two albums were released first?
        <A4>: Indefinite
        <Q5>: {question}
        <A5>:
        """

        return prompt


def main():
    labeller = QuestionLabeller()
    file_handler = FileHandler()

    dataset_name = "Squad_Dummy"

    questions = list(file_handler.load_df(dir = "DataFiles", dataset_name = dataset_name, filename = "queries.csv")["query"])
    labels = labeller.label(questions)

    df = pd.DataFrame({"question" : questions, "label" : labels})

    file_handler.save_df(df = df, dir = "RetrievalFiles", dataset_name = dataset_name, 
                         filename = "question_labels.csv", header = True)

if __name__ == "__main__":
    main()