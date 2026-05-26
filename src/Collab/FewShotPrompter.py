'''
    Code has been taken from my Msc Thesis Code repo, with the initial experiment
'''

# https://discuss.huggingface.co/t/llama-2-7b-hf-repeats-context-of-question-directly-from-input-prompt-cuts-off-with-newlines/48250/6
'''
    A class for constructing prompts for the LLM.
'''
class FewShotPrompter:
    def __init__(self):

        self.few_shots_QA = ["\n<Q1>: Who won wimbledon 2015?\n<A1>: Novak Djokovic",
                            "\n<Q2>: Does the sun have temperatures excedding 100 degrees?\n<A2>: yes",
                            "\n<Q3>: What is the capital of Italy?\n<A3>: Rome",
                            "\n<Q4>: Have Novak Djokovic or Roger Federer won more grandslams?\n<A4>: Novak Djokovic",
                            "\n<Q5>: What does LLM stand for?\n<A5>: Large Language Model",
                            "\n<Q6>: Is it true, that the atom representing oxygen, is number 14 in the periodic table?\n<A6>: no"
        ]

        self.few_shots_QCA = ["\n<Q1>: Who won wimbledon 2015?\n<C1>: The final of wimbledon 2015, was played between federer and Djokovic. Novak Djokovic won\n<A1>: Novak Djokovic",
                            "\n<Q2>: Does the sun have temperatures excedding 100 degrees?\n<C2>: The sun is very hot, with temperatures much higher than those observed in normal life\n<A2>: yes",
                            "\n<Q3>: What is the capital of Italy?\n<C3>: The capital of Italy, a country in Europe, is Rome.\n<A3>: Rome",
                            "\n<Q4>: Have Novak Djokovic or Roger Federer won more grandslams?\n<C4>: Djokovic, a tennis player from Serbia, has won 24 grand slams. One of his well known rivals, Federer, has won 20\n<A4>: Novak Djokovic",
                            "\n<Q5>: What does LLM stand for?\n<C5>: An LLM, is a probabilitic model, called a Large Language Model.\n<A5>: Large Language Model",
                            "\n<Q6>: Is it true, that the atom representing oxygen, is number 14 in the periodic table?\n<C6>: Oxygen, a very well known atom crucial to the existence of humans, has number 8 in the periodic table\n<A6>: no"
        ]
    
    ''' 
        Constructs a prompt from the question and context:

        question: A string
        context: A string

        Returns: The prompt as a string
    '''
    def construct_prompt_with_context(self, question, context):
        prompt = "Answer the question <Q7>, by generating ONLY the answer <A7>, based on information in <C7>. If the context does not provide sufficient information to answer the question, you must answer: IDK"

        for shot in self.few_shots_QCA:
            prompt += shot 
        
        prompt += "\n<Q7>: {}\n<C7>: {}\n<A7>:".format(question, context)
        return prompt
    
    def construct_safe_prompt_with_context(self, question, context):
        prompt = """Answer the question <Q7>, by generating ONLY the answer <A7>, based on information in <C7>.
        If the context provides sufficient information to answer the question, then simply answer the question.
        If the context does not provide sufficient information to answer the question, answer: IDK
        If the context provides context that is wrong, but you dont know the correct answer, answer: FI
        If the context provides context that is wrong, and you know the correct answer by your internal knowledge, answer: FI_ans, where ans is the correct answer"""

        for shot in self.few_shots_QCA:
            prompt += shot 
        
        prompt += "\n<Q7>: {}\n<C7>: {}\n<A7>:".format(question, context)
        return prompt

    '''
        Constructs a prompt from only the question.

        question: A string

        Returns: The prompt as a string
    '''
    def construct_prompt_without_context(self, question):
        prompt = "Answer the question <Q7>, by generating ONLY the answer <A7>. If you dont know the correct answer, you must answer: IDK"

        for shot in self.few_shots_QA:
            prompt += shot 
        
        prompt += "\n<Q7>: {}\n<A7>:".format(question)
        return prompt
    
    '''
        Constructs prompts from a list of questions and contexts. Lists of questions and contexts, 
        contains corresponding questions and contexts as strings
    '''
    def construct_prompts_with_context(self, questions, contexts):
        return [self.construct_prompt_with_context(question, context) for (question, context) in zip(questions, contexts)]
    
    def construct_safe_prompts_with_context(self, questions, contexts):
        return [self.construct_safe_prompt_with_context(question, context) for (question, context) in zip(questions, contexts)]
    
    '''
        Constructs prompts from a list of questions. questions is a list of strings.
    '''
    def construct_prompts_without_context(self, questions):
        return [self.construct_prompt_without_context(question) for question in questions]