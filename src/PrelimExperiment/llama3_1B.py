

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Llama1B:

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device = {}".format(self.device))

        # https://discuss.huggingface.co/t/the-effect-of-padding-side/67188
        checkpoint = "meta-llama/Llama-3.2-1B"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map = "cuda", padding_side = "left")
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map = "cuda")
        self.model.generation_config.do_sample = False

        # Add a padding token
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        #self.tokenizer.pad_token = self.tokenizer.eos_token
        # https://stackoverflow.com/questions/68166721/pytorch-fails-with-cuda-error-device-side-assert-triggered-on-colab
        self.model.resize_token_embeddings(len(self.tokenizer))

    '''
        Returns answers for prompts. Note, a prompt can be a question combined
        with context or just the question. 
    '''
    # https://huggingface.co/docs/transformers/main_classes/text_generation
    def answer(self, prompts):
        # Returns input_ids and attention_mask. Pads to the longest prompt
        inputs = self.tokenizer(prompts, return_tensors = "pt", padding = "longest")
        inputs = inputs.to(self.device)
        # https://github.com/huggingface/transformers/issues/12503
        generate_ids = self.model.generate(**inputs, max_new_tokens = 75)
        answers = self.tokenizer.batch_decode(generate_ids, skip_special_tokens = True)

        answers_extracted = [self._extract_answer(answer) for answer in answers]

        return answers, answers_extracted
    
    
    # https://www.geeksforgeeks.org/python/python-extract-string-between-two-substrings/
    # https://www.w3schools.com/python/ref_string_split.asp
    def _extract_answer(self, answer):
        try:
            _, p2 = answer.split("<A4>:", 1)
            p3, _ = p2.split("\n", 1)
            return p3
        except:
            p1, p2 = answer.split("<A4>:", 1)
            return p2

