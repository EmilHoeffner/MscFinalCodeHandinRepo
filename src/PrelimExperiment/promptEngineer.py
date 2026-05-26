
# https://discuss.huggingface.co/t/llama-2-7b-hf-repeats-context-of-question-directly-from-input-prompt-cuts-off-with-newlines/48250/6
'''
    A class for constructing prompts for the LLM.
'''
class PromptEngineer:
    def __init__(self):
        self.few_shots_QA = ["\n<Q1>: Who won wimbledon 2015?\n<A1>: Novak Djokovic",
        "\n<Q2>: What is the capital of Italy?\n<A2>: Rome",
        "\n<Q3>: What does LLM stand for?\n<A3>: Large Language Model"]

        self.few_shots_QCA = ["\n<Q1>: Who won wimbledon 2015?\n<C1>: The final of wimbledon 2015, was played between federer and Djokovic. Novak Djokovic won\n<A1>: Novak Djokovic",
        "\n<Q2>: What is the capital of Italy?\n<C2>: The capital of Italy, a country in Europe is Rome.\n<A2>: Rome",
        "\n<Q3>: What does LLM stand for?\n<C3>: An LLM, is a probabilitic model, called a Large Language Model.\n<A3>: Large Language Model"]
    
    ''' 
        Constructs a prompt from the question and context:

        question: A string
        context: A string

        Returns: The answer as a string.
    '''
    def construct_prompt_with_context(self, question, context):
        prompt = "Answer the question <Q4>, by generating ONLY the answer <A4>. [DO NOT REPEAT THE INPUT PROMPT]"

        for shot in self.few_shots_QCA:
            prompt += shot 
        
        prompt += "\n<Q4>: {}\n<C4>: {}\n<A4>:".format(question, context)
        return prompt

    
    
    def construct_prompt_without_context(self, question):
        prompt = "Answer the question <Q4>, by generating ONLY the answer <A4>. [DO NOT REPEAT THE INPUT PROMPT]"

        for shot in self.few_shots_QA:
            prompt += shot 
        
        prompt += "\n<Q4>: {}\n<A4>:".format(question)
        return prompt
    
    def construct_prompts_with_context(self, questions, contexts):
        return [self.construct_prompt_with_context(question, context) for (question, context) in zip(questions, contexts)]
    
    def construct_prompts_without_context(self, questions):
        return [self.construct_prompt_without_context(question) for question in questions]