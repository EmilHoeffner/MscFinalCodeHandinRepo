
import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import sys

from tqdm import tqdm

sys.path.append("..")

from Constants import constants

# https://huggingface.co/google/flan-t5-xxl/discussions/41
# https://github.com/huggingface/transformers/issues/5204
# https://community.deeplearning.ai/t/flant5-maximum-input-and-output-length/421273
# https://huggingface.co/docs/transformers/main/en/model_doc/t5#transformers.T5Config
# https://discuss.huggingface.co/t/passing-inputs-longer-than-512-tokens-after-pretraining-a-t5-model-is-it-safe/170655/3

class FlanT5:

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device = {}".format(self.device))
        
        #self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl", device_map = "cuda")
        #self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl", device_map = "cuda")

        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
        
        self.tokenizer.model_max_length = constants.tokenizer_max_length()
        self.model.generation_config.do_sample = False
        self.PAD_ID = self.tokenizer.pad_token_id
        assert(self.PAD_ID == 0)


    # https://huggingface.co/docs/transformers/main_classes/text_generation
    # https://discuss.huggingface.co/t/how-can-i-obtain-the-logits-via-model-generate/110636/2
    # https://www.kaggle.com/discussions/questions-and-answers/557651
    # https://discuss.pytorch.org/t/convert-a-tuple-into-tensor/82964/2
    # https://huggingface.co/google/gemma-2b-it/discussions/55
    # https://www.kaggle.com/discussions/questions-and-answers/557651
    def answer(self, prompts):
        # Returns input_ids and attention_mask. Pads to the longest prompt
        inputs = self.tokenizer(prompts, return_tensors = "pt", padding = "longest")
        #inputs = inputs.to(self.device)

        num_new_tokens = 40
        # https://github.com/huggingface/transformers/issues/12503
        # https://discuss.huggingface.co/t/output-dimension-of-automodelforcausallm/47225
        # https://huggingface.co/docs/transformers/internal/generation_utils
        # https://huggingface.co/TheBloke/Llama-2-70B-Chat-GPTQ/discussions/25
        outputs = self.model.generate(**inputs, max_new_tokens = num_new_tokens, output_logits = True, return_dict_in_generate = True, do_sample = False)
        # Output.sequences has shape [batch_size, num_input_tokens + num_generated_tokens]
        generated_tokens = outputs.sequences
        
        answer_lengths = []
        B = generated_tokens.shape[0]
        for i in range(B):
            tokens = generated_tokens[i, :]
            tokenids_stripped = [token_id for token_id in tokens if token_id != self.PAD_ID]
            answer_lengths.append(len(tokenids_stripped))


        answers = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens = True)
        #answers = [answer.strip() for answer in answers]
        # Converts tuple of (batch_size, vocab_size) to (num_generated_tokens, batch_size, vocab_size)
        logits = self.tuple_of_tensors_to_tensor(outputs.logits)

        #print("Generated Tokens Shape = {}".format(generated_tokens.shape))
        #print("Logits shape = {}".format(logits.shape))

        return answers, logits.cpu().detach(), answer_lengths


    # https://discuss.huggingface.co/t/how-to-get-cross-attention-values-of-t5/970/3
    # https://discuss.huggingface.co/t/what-the-tokens-are-cross-attentions-output-for/14420
    # https://huggingface.co/docs/transformers/v4.26.1/model_doc/t5
    # https://discuss.huggingface.co/t/t5-why-do-we-have-more-tokens-expressed-via-cross-attentions-than-the-decoded-sequence/31893
    # https://huggingface.co/docs/transformers/v4.13.0/model_doc/t5
    def cross_attention_map(self, prompts):
        # Returns input_ids and attention_mask. Pads to the longest prompt
        num_new_tokens = 20
        # https://github.com/huggingface/transformers/issues/12503
        # https://discuss.huggingface.co/t/output-dimension-of-automodelforcausallm/47225
        # https://huggingface.co/docs/transformers/internal/generation_utils
        # https://huggingface.co/TheBloke/Llama-2-70B-Chat-GPTQ/discussions/25
        

        answers = []
        output_attentions = []
        inputs_tokens = []

        for prompt in tqdm(prompts):
            input = self.tokenizer(prompt, return_tensors = "pt")
            input_tokens = self.tokenizer.tokenize(prompt, return_tensors = "pt")
            inputs_tokens.append(input_tokens)

            #input = input.to(self.device)
            output = self.model.generate(**input, max_new_tokens = num_new_tokens, output_attentions = True, return_dict_in_generate = True, do_sample = False)
            generated_tokens = output.sequences
            cross_attention_map = output.cross_attentions

            answer = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens = True)[0]


            answers.append(answer)
            output_attentions.append(cross_attention_map)

        return answers, output_attentions, inputs_tokens
    
    '''
        sentence: A string

        Tokenizes the sentence and returns the result
    '''
    def tokenize(self, sentence):
        return self.tokenizer(sentence, return_tensors = "pt")

    
    # https://discuss.pytorch.org/t/convert-a-tuple-into-tensor/82964/2
    def tuple_of_tensors_to_tensor(self, tuple_of_tensors):
        return torch.stack(list(tuple_of_tensors), dim=0)
    

def main():
    m = FlanT5()


if __name__ == "__main__":
    main()