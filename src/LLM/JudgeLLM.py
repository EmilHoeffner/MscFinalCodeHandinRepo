# https://huggingface.co/BAAI/JudgeLM-7B-v1.0

# https://github.com/baaivision/JudgeLM/tree/main/judgelm/llm_judge

# https://huggingface.co/Unbabel/M-Prometheus-3B?library=transformers

from transformers import AutoTokenizer, AutoModelForCausalLM

class JudgeLLM:

    def __init__(self):        
        self.tokenizer = AutoTokenizer.from_pretrained("Unbabel/M-Prometheus-3B", padding_side = "left")
        self.model = AutoModelForCausalLM.from_pretrained("Unbabel/M-Prometheus-3B")
        self.model.generation_config.do_sample = False

    
    '''
        questions: List of questions
        answers: List of answers

        Returns: List of the Judge predictions of answer quality with tuples (explanation : string, rating : int)
    '''
    def answer(self, questions, answers):
        prompt = [self.to_prompt(question, answer) for (question, answer) in zip(questions, answers)]
        num_new_tokens = 120
        inputs = self.tokenizer(prompt, return_tensors = "pt", padding = "longest")
        outputs = self.model.generate(**inputs, max_new_tokens = num_new_tokens, do_sample = False)

        input_length = len(inputs["input_ids"][0])
        # Output.sequences has shape [batch_size, num_input_tokens + num_generated_tokens]
        generated_tokens = outputs[:, input_length:]

        answers = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens = True)

        return [self.parse_answer(answer) for answer in answers]

    
    '''
        answer: String with answer of the LLM.
    '''
    def parse_answer(self, answer):
        # Extract from <E6>
        try:
            s = answer.split("<E6>:")[1]
            split = s.split("<R6>:")
            explanation = split[0].strip()
            s = split[1]
            rating = int(s.split("\n")[0].strip())
            return explanation, rating
        except:
            raise Exception(f"LLM Judge did not answer in correct format. The answer:\n{answer}")



    '''
        question: A single string, containing the question.
        answer: A single string, containing the answer.

        Returns: A prompt suitable to feed the judge.
    '''
    # Prompt modified from https://huggingface.co/learn/cookbook/llm_judge
    def to_prompt(self, question, answer):
        prompt = f"""
        You will be given a user_question and system_answer couple.
        Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question.
        Give your answer on a scale of 1 to 4, where 1 means that the system_answer is not helpful at all, and 4 means that the system_answer completely and helpfully addresses the user_question.

        Here is the scale you should use to build your answer:
        1: The system_answer is terrible: completely irrelevant to the question asked, or very partial or incorrect
        2: The system_answer is mostly not helpful: misses some key aspects of the question
        3: The system_answer is mostly helpful: provides support, but still could be improved
        4: The system_answer is excellent: relevant, direct, detailed, and addresses all the concerns raised in the question

        Provide your feedback as follows:

        Feedback:::
        <E6>: (your rationale for the rating, as a text)
        <R6>: (your rating, as a number between 1 and 4)

        You MUST provide values for '<E6>:' and '<R6>:' in your answer. DO NOT REPEAT QUESTIONS OR ANSWERS
        Keep the evaluation brief (you have maximum of 40 tokens to answer).

        Given these below 5 examples, please provide explanation and rating for the 6'th example

        <Q1>: Is labrador a type of dog
        <A1>: Yes, Labrador is a dog breed
        <E1>: The answer is good and concise
        <R1>: 4
        <Q2>: Does a computer use electricity?
        <A2>: A computer is an electronic device
        <E2>: The Answer is overall good, but does not answer the question directly
        <R2>: 3
        <Q3>: Does water contain H20 molecules?
        <A3>: Water contains molecules
        <E3>: The answer is related to the question, but does not answer it
        <R3>: 2
        <Q4>: Are clouds white
        <A4>: blue is a color
        <E4>: The answer is useless and unrelated to the question
        <R4>: 1
        <Q5>: When was Roger Federer born?
        <A5>: Roger Federer was born in 1755
        <E5>: The answer is incorrect 
        <R5>: 1
        <Q6>: {question}
        <A6>: {answer}
        """

        return prompt


