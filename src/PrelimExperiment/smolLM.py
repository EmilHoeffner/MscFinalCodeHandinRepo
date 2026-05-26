# https://huggingface.co/HuggingFaceTB/SmolLM-1.7B?library=transformers
# https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaForCausalLM
# https://stackoverflow.com/questions/77237818/how-to-load-a-huggingface-pretrained-transformer-model-directly-to-gpu
# https://discuss.huggingface.co/t/sending-a-dataset-or-datasetdict-to-a-gpu/17208/2

# https://github.com/huggingface/transformers/issues/14521

# https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B

# https://huggingface.co/HuggingFaceTB/SmolLM3-3B

# https://discuss.huggingface.co/t/llama-2-repeats-its-prompt-as-output-without-answering-the-prompt/78230/2

# https://discuss.huggingface.co/t/repetition-issues-in-llama-models-3-8b-3-70b-3-1-3-2/144196

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class SmolLM:

    def __init__(self, version):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device = {}".format(self.device))

        # https://discuss.huggingface.co/t/the-effect-of-padding-side/67188
        if version == 1:
            checkpoint = "HuggingFaceTB/SmolLM-1.7B"
        elif version == 2:
            checkpoint = "HuggingFaceTB/SmolLM2-1.7B"
        elif version == 3:
            checkpoint = "HuggingFaceTB/SmolLM3-3B"
        else:
            raise Exception("Non Valid SmolLM version chosen")

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map = "cuda", padding_side = "left")
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map = "cuda")
        self.model.generation_config.do_sample = False

        # Add a padding token
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
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
        generate_ids = self.model.generate(**inputs, max_new_tokens = 75)#, stop_strings = ["[EA4]"], tokenizer = self.tokenizer)
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

