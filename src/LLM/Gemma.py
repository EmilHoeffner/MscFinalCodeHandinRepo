'''
    Code has been taken from my initial Msc Thesis Code repo (with the pre-thesis experiment)
'''


# https://huggingface.co/google/gemma-2-2b-it

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# https://discuss.huggingface.co/t/inf-values-for-logit-score-outputs-with-model-generate/67108
class Gemma:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device = {}".format(self.device))

        self.use_test_model = False

        if self.use_test_model:
            checkpoint = "google/gemma-3-1b-it"
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map = "cuda", padding_side = "left")
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map = "cuda")
        else:
            checkpoint = "google/gemma-2-2b-it"
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side = "left")
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint)

        self.PAD_ID = self.tokenizer.pad_token_id
        assert(self.PAD_ID == 0)

    '''
        Returns answers for prompts. Note, a prompt can be a question combined
        with context or just the question. 
        
        Prompts is a list of strings.

        Returns the full answers, as well as the answers, with only the answer extracted. 
    '''
    # https://huggingface.co/docs/transformers/main_classes/text_generation
    # https://discuss.huggingface.co/t/how-can-i-obtain-the-logits-via-model-generate/110636/2
    # https://www.kaggle.com/discussions/questions-and-answers/557651
    # https://discuss.pytorch.org/t/convert-a-tuple-into-tensor/82964/2
    # https://huggingface.co/google/gemma-2b-it/discussions/55
    # https://www.kaggle.com/discussions/questions-and-answers/557651
    def answer(self, prompts):
        # Returns input_ids and attention_mask. Pads to the longest prompt
        #messages = [[{f"role" : "user", "content" : prompt}] for prompt in prompts]
        #inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt", padding = "longest", return_dict=True)

        inputs = self.tokenizer(prompts, return_tensors = "pt", padding = "longest")

        if self.use_test_model:
            inputs = inputs.to(self.device)
            
        num_new_tokens = 40
        # https://github.com/huggingface/transformers/issues/12503
        # https://discuss.huggingface.co/t/output-dimension-of-automodelforcausallm/47225
        # https://huggingface.co/docs/transformers/internal/generation_utils
        # https://huggingface.co/TheBloke/Llama-2-70B-Chat-GPTQ/discussions/25
        outputs = self.model.generate(**inputs, max_new_tokens = num_new_tokens, output_logits = True, return_dict_in_generate = True, do_sample = False)
        inputs_length = len(inputs["input_ids"][0])
        # Output.sequences has shape [batch_size, num_input_tokens + num_generated_tokens]
        generated_tokens = outputs.sequences[:, inputs_length:]
        
        answer_lengths = []
        B = generated_tokens.shape[0]
        for i in range(B):
            tokens = generated_tokens[i, :]
            tokenids_stripped = [token_id for token_id in tokens if token_id != self.PAD_ID]
            answer_lengths.append(len(tokenids_stripped))

        answers = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens = True)
        answers = [answer.strip() for answer in answers]
        # Converts tuple of (batch_size, vocab_size) to (num_generated_tokens, batch_size, vocab_size)
        logits = self.tuple_of_tensors_to_tensor(outputs.logits)

        #print("Generated Tokens Shape = {}".format(generated_tokens.shape))
        #print("Logits shape = {}".format(logits.shape))

        return answers, logits.cpu().detach(), answer_lengths

    '''
        sentence: A string

        Tokenizes the sentence and returns the result
    '''
    def tokenize(self, sentence):
        return self.tokenizer(sentence, return_tensors = "pt")

    
    # https://discuss.pytorch.org/t/convert-a-tuple-into-tensor/82964/2
    def tuple_of_tensors_to_tensor(self, tuple_of_tensors):
        return torch.stack(list(tuple_of_tensors), dim=0)