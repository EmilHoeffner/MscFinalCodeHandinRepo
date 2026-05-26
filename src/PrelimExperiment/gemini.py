# https://ai.google.dev/gemini-api/docs/quickstart

# https://ai.google.dev/gemini-api/docs/api-key#provide-api-key-explicitly

# https://aistudio.google.com/prompts/new_chat?_gl=1*1phaxez*_ga*MTU1MTE2MDUxOS4xNzU5MzEwMDYx*_ga_P1DBVKWT6V*czE3NTk3Mzk3MzIkbzEkZzEkdDE3NTk3Mzk3NDgkajQ0JGwwJGgxOTA5NjgxOTU.

from google import genai
import keymanagement

class GeminiAPI:
    def __init__(self):
        api_key = keymanagement.get_gemini_API_key()
        self.client = genai.Client(api_key = api_key)

    def answer(self, prompts):
        model = "gemini-2.5-flash"
        answers = []

        try:
            for prompt in prompts:
                generated_answer = self.client.models.generate_content(
                model = model, contents = prompt)
                answer_text = generated_answer.text 
                answers.append(answer_text)
        except:
            print("Answered {} prompts before excedding quota".format(len(answers)))
            return answers
        
        return answers