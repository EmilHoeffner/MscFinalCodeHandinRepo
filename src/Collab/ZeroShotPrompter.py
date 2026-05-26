
'''
    Code has been taken from my Msc Thesis Code repo, with the initial experiment
'''

# https://discuss.huggingface.co/t/llama-2-7b-hf-repeats-context-of-question-directly-from-input-prompt-cuts-off-with-newlines/48250/6
'''
    A class for constructing zero shot prompts for the LLM.
'''
class ZeroShotPrompter:
    def __init__(self):
        pass
    
    ''' 
        Constructs a prompt from the question and context:

        question: A string
        context: A string

        Returns: The prompt as a string
    '''
    def construct_prompt_with_context(self, question, context):
        prompt = "Answer the question <Q>, by generating ONLY the answer <A>, based on information in <C>. If the context does not provide sufficient information to answer the question, you must answer: IDK"
        prompt += "\n<Q>: {}\n<C>: {}\n<A>:".format(question, context)
        return prompt
    
    def construct_safe_prompt_with_context(self, question, context):
        prompt = """Answer the question <Q>, by generating ONLY the answer <A>, based on information in <C>.
        If the context provides sufficient information to answer the question, then simply answer the question.
        If the context does not provide sufficient information to answer the question, answer: IDK
        If the context provides context that is wrong, but you dont know the correct answer, answer: FI
        If the context provides context that is wrong, and you know the correct answer by your internal knowledge, answer: FI_ans, where ans is the correct answer"""
        
        prompt += "\n<Q>: {}\n<C>: {}\n<A>:".format(question, context)
        return prompt

    '''
        Constructs a prompt from only the question.

        question: A string

        Returns: The prompt as a string
    '''
    def construct_prompt_without_context(self, question):
        prompt = "Answer the question <Q>, by generating ONLY the answer <A>. If you dont know the correct answer, you must answer: IDK"
        
        prompt += "\n<Q>: {}\n<A>:".format(question)
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