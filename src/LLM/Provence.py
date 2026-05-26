
import nltk

from transformers import AutoModel

# https://huggingface.co/naver/provence-reranker-debertav3-v1

# https://huggingface.co/BlueOrangeDigital/distilbert-cross-segment-document-chunking

import sys 
import numpy as np

sys.path.append("..")

from Datasets.WikiMultiHop import WikiMultiHop
from Datasets.Squad import Squad


from transformers import (
    AutoModelForSequenceClassification,
    DistilBertTokenizer,
    TextClassificationPipeline
)

class Provence:

    def __init__(self):
        nltk.download('punkt_tab') 
        self.provence = AutoModel.from_pretrained("naver/provence-reranker-debertav3-v1", trust_remote_code = True)

        chunk_model_name = "BlueOrangeDigital/distilbert-cross-segment-document-chunking"
        id2label = {0: "SAME", 1: "DIFFERENT"}
        label2id = {"SAME": 0, "DIFFERENT": 1}

        self.chunk_tokenizer = DistilBertTokenizer.from_pretrained(chunk_model_name)
        self.chunk_model = AutoModelForSequenceClassification.from_pretrained(
            chunk_model_name,
            num_labels=2,
            id2label=id2label,
            label2id=label2id
        )

    def prune_directly(self, question, context, threshold = 0.1):
        try:
            provence_output = self.provence.process([question], [[context]], threshold = threshold)
            pruned_context = provence_output["pruned_context"][0]
            
            return pruned_context[0]
        except:
            return context


    def weigh_sentences(self, question, context, top_k_prop = 0.2):
        try:
            context_split = context.split(".")
            N_splits = len(context_split)
            top_k = max(1, int(N_splits * top_k_prop))
            provence_output = self.provence.process([question], [context_split], threshold = 0.1)
            rank_scores = np.array(provence_output["reranking_score"][0])
            N_ranks = len(rank_scores)

            assert(N_splits == N_ranks)

            if rank_scores[-1] is None:
                rank_scores = rank_scores[0:N_ranks - 1]
                context_split = context_split[0:N_ranks - 1]

            rank_arg_sort = np.argsort(rank_scores)[::-1]
            rank_arg_sort = rank_arg_sort[0:top_k]

            c = ""

            for i,context in enumerate(context_split):
                if i in rank_arg_sort:
                    c += f"[{context}]. "
                else:
                    c += f"{context}. "
            
            return c
        except:
            print("Error in Weighting")
            return context


    def prune_sentences(self, question, context, threshold = 0.1):

        try:
            provence_output = self.provence.process([question], [context.split(".")], threshold = threshold)
            pruned_context = provence_output["pruned_context"][0]
            
            c = ""

            for i,context in enumerate(pruned_context):
                if context != "":
                    c += context
                
                if context != "" and i < len(pruned_context) - 1:
                    c += ". "
                elif context == "" and i == len(pruned_context) - 1:
                    c += "."
            
            return c
        except:
            return context
    
    def prune_chunks(self, question, context, threshold = 0.01):
        # https://stackoverflow.com/questions/63853155/separate-string-every-3-characters-and-putting-them-in-a-list
        context_splits = context.split(".")

        pipe = TextClassificationPipeline(model=self.chunk_model, tokenizer=self.chunk_tokenizer, top_k = None)
        new_context_split = []

        current_passage = context_splits[0]
        
        for i in range(1, len(context_splits)):
            context_to_merge = context_splits[i]
            pair = current_passage + " [SEP] " + context_to_merge
            res = pipe(pair)[0]
            same_score = res[0]["score"]
            different_score = res[1]["score"]

            # To avoid merging if exceds 512 tokens.
            if same_score > different_score and len(current_passage + ". " + context_to_merge) < 1000:
                current_passage += ". " + context_to_merge
                if i == len(context_splits) - 1:
                    new_context_split.append(current_passage)
            else:
                new_context_split.append(current_passage)

                if i == len(context_splits) - 1:
                    new_context_split.append(". " + context_to_merge)
                else:
                    current_passage = context_to_merge

        # Add original contexts without pruning them individually. 
        try:
            provence_output = self.provence.process([question], [new_context_split], threshold = threshold)
            pruned_context = provence_output["pruned_context"][0]
            
            c = ""

            for i,pruned_context in enumerate(pruned_context):
                if pruned_context != "":
                    c += new_context_split[i]
            return c
        except:
            return context

        
    

def main():
    pruner = Provence()
    dataset = WikiMultiHop("Tune")
    #dataset = Squad("Tune")

    i = 1920

    question, context, answer, qid = dataset[i]


    weighted_context = pruner.weigh_sentences(question, context)

    print("Question = {}".format(question))
    print("\nCONTEXT = \n{}".format(context))
    print("\nPRUNED_CONTEXT = \n{}".format(weighted_context))



if __name__ == "__main__":
    main()