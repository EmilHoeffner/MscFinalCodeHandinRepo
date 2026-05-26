# https://medium.com/@yashpaddalwar/how-to-access-free-open-source-llms-like-llama-3-from-hugging-face-using-python-api-step-by-step-5da80c98f4e3

# https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
# https://github.com/huggingface/huggingface_hub/issues/2586

# For running LLama 3 with API: https://colab.research.google.com/#fileId=https%3A//huggingface.co/meta-llama/Llama-3.1-8B-Instruct.ipynb

# https://huggingface.co/meta-llama/Llama-3.1-8B

import os
from openai import OpenAI

from huggingface_hub import login

import keymanagement

from transformers import pipeline

class Llama3API:

    def __init__(self):
        os.environ['HF_TOKEN'] = keymanagement.get_huggingface_API_key()

        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ["HF_TOKEN"],
        )

        
    def _answer_prompt(self, query):
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        # Below model_name does not work, since it is not conversational.
        # model_name = "meta-llama/Llama-3.1-8B"
        completion = self.client.chat.completions.create(
            model = model_name,
            messages = [
                {
                    "role": "user",
                    "content": query
                }
            ],
        )

        return completion.choices[0].message.content
    
    def answer(self, prompts):
        answers = []

        for prompt in prompts:
            answers.append(self._answer_prompt(prompt))
        
        return answers

class Llama3Local:

    def __init__(self):
        model_name = "meta-llama/Llama-3.1-8B"
        login(new_session=False, token = keymanagement.get_huggingface_API_key())
        self.pipe = pipeline("text-generation", model = model_name)
        
    def answer_prompt(self, query):
        pass


