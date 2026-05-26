
'''
    Code has been taken from my Msc Thesis Code repo, with the initial experiment
'''

# https://discuss.huggingface.co/t/llama-2-7b-hf-repeats-context-of-question-directly-from-input-prompt-cuts-off-with-newlines/48250/6
'''
    A class for constructing zero shot prompts for the LLM.
'''
class WeightPrompter:
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
        prompt += "Sentences enclosed in square brackets [] are likely to be of high importance for answering the question. It is thus advised to attend highly to such sentences"
        prompt += "\n<Q>: {}\n<C>: {}\n<A>:".format(question, context)
        return prompt
    
    '''
        Constructs prompts from a list of questions and contexts. Lists of questions and contexts, 
        contains corresponding questions and contexts as strings
    '''
    def construct_prompts_with_context(self, questions, contexts):
        return [self.construct_prompt_with_context(question, context) for (question, context) in zip(questions, contexts)]