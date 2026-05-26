
import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# https://huggingface.co/google/flan-t5-xxl/discussions/41
# https://github.com/huggingface/transformers/issues/5204
# https://community.deeplearning.ai/t/flant5-maximum-input-and-output-length/421273

class FlanT5:

    def __init__(self, xl = False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device = {}".format(self.device))

        if xl == True:
            self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl", device_map = "cuda")
            self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl", device_map = "cuda")
            self.tokenizer.model_max_length = 4096
            self.model.generation_config.do_sample = False
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", device_map = "cuda")
            self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large", device_map = "cuda")
            self.tokenizer.model_max_length = 4096
            self.model.generation_config.do_sample = False

    def answer(self, prompts):
        inputs = self.tokenizer(prompts, return_tensors = "pt", padding = "longest")
        inputs = inputs.to(self.device)
        # https://github.com/huggingface/transformers/issues/12503
        generate_ids = self.model.generate(**inputs, max_new_tokens = 75)#, stop_strings = ["[EA4]"], tokenizer = self.tokenizer)
        answers = self.tokenizer.batch_decode(generate_ids, skip_special_tokens = True)

        return ["blank" for a in answers], answers