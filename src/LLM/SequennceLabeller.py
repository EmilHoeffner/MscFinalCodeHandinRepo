# https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForTokenClassification

# https://huggingface.co/docs/transformers/main/training
# https://huggingface.co/docs/transformers/main/training

from transformers import AutoTokenizer, BertForTokenClassification, TrainingArguments, Trainer
import torch
import numpy as np

import sys 

sys.path.append("..")

from Datasets.WikiMultiHop import WikiMultiHop

class SequenceLabeller_Dataset:
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids 
        self.attention_masks = attention_masks
        self.labels = labels

        self.N = len(self.input_ids)        

    def __len__(self):
        return self.N
    
    def __getitem__(self, index):
        return {"input_ids" : self.input_ids[index],  "attention_mask" : self.attention_masks[index], "labels" : self.labels[index]}


class SequenceLabeller:

    def __init__(self, load_pretrained):
        self.path = "SeqLabeller"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", device_map = "cuda")

        if load_pretrained:
            self.model = BertForTokenClassification.from_pretrained(self.path, num_labels = 2, device_map = "cuda")
        else:
            self.model = BertForTokenClassification.from_pretrained("google-bert/bert-base-uncased", num_labels = 2, device_map = "cuda")

    
    '''
        Passages: List of lists, where each entry holds a list of passages.
        Labels: List of lists, where each entry holds a label saying if the passage is relevant or not.
    '''
    def ready_data(self, passages_sentences, labels_sentences):
        out_input_ids = []
        out_labels = []
        out_attention_masks = []

        for sample_passages, sample_labels in zip(passages_sentences, labels_sentences):
            token_ids = []
            token_attention_mask = []
            token_labels = []

            for passage, label in zip(sample_passages, sample_labels):
                tokeniser_data = self.tokenizer(passage)
                input_ids = tokeniser_data["input_ids"]
                attention_mask = tokeniser_data["attention_mask"]
                labels = [label for i in range(len(input_ids))]

                token_ids += input_ids
                token_labels += labels 
                token_attention_mask += attention_mask

            
            assert(len(token_ids) == len(token_attention_mask))
            assert(len(token_attention_mask) == len(token_labels))

            out_input_ids.append(token_ids)
            out_labels.append(token_labels)
            out_attention_masks.append(token_attention_mask)

        PAD_token_id = self.tokenizer.pad_token_id

        lengths = [len(l) for l in out_input_ids]
        max_length = int(np.max(np.array(lengths)))

        out_input_ids = [out_input_ids[i] + (max_length - l) * [PAD_token_id] for i,l in enumerate(lengths)]
        out_labels = [out_labels[i] + (max_length - l) * [0] for i,l in enumerate(lengths)]
        out_attention_masks = [out_attention_masks[i] + (max_length - l) * [0] for i,l in enumerate(lengths)]

        #return {"input_ids" : out_input_ids, "attention_mask" : out_attention_masks, "labels" : out_labels}
        return SequenceLabeller_Dataset(out_input_ids, out_attention_masks, out_labels)

        
    
    def train(self, passages_sentences_train, passages_labels_train, passages_sentences_val, passages_labels_val): 
        train_data = self.ready_data(passages_sentences_train, passages_labels_train)
        val_data = self.ready_data(passages_sentences_val, passages_labels_val)

        val_data = self.ready_data(passages_sentences_val, passages_labels_val)
        training_args = TrainingArguments(
            output_dir=self.path,
            num_train_epochs=3,
            per_device_train_batch_size=32,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            bf16=True,
            learning_rate=2e-5,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        ) 

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            )

        trainer.train()

    '''
        sentences: A list of strings
    '''
    def predict(self, sentences):
        pass



def main():
    wiki = WikiMultiHop("Tune")
    '''
    labeller = SequenceLabeller(load_pretrained = False)   

    sentences = [["The house was brown", "The house was yellow and blue"], ["mouse", "Cow, Cow, Cow"]] 
    labels = [[1, 0], [1, 1]]

    labeller.train(sentences, labels, sentences, labels)
    '''

if __name__ == "__main__":
    main()