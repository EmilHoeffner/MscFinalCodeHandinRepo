'''
    Code has been taken from my Msc Thesis Code repo, with the pre-thesis experiment
'''

# https://www.geeksforgeeks.org/nlp/tokenize-text-using-nltk-python/

import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import time
import sys
import pandas as pd
from tqdm import tqdm 

import torch
from bert_score import BERTScorer
from torchmetrics.text import CHRFScore

import Levenshtein

sys.path.append("..")

from listutils import list_distinct


class Evaluater:
    def __init__(self, load_scorers = True, character_overlap = 6, n_gram_overlap = 2):
        nltk.download('punkt')

        if load_scorers:
            self.BERT_scorer = BERTScorer(model_type = 'bert-base-uncased')
            # Make invariant to casing and whitespace 
            self.chrf_scorer = CHRFScore(lowercase = False, whitespace = False, n_char_order = character_overlap, 
                                         n_word_order = n_gram_overlap)

        self.loaded_lettuce = False 
        self.loaded_hallucinot = False
        self.loaded_judge = False

    def LLM_As_Judge(self, qids, questions, predicted_answers, true_answers, sample_size = 10):
        # Do the LLM as Judge evaluation

        if not(self.loaded_judge):
            self.judge = JudgeLLM()
            self.loaded_judge = True

        np.random.seed(42)
        sample_indices = np.random.choice(a = len(questions), size = sample_size, replace = False)

        QI = [qids[index] for index in sample_indices]
        Q = [questions[index] for index in sample_indices]
        AP = [predicted_answers[index] for index in sample_indices]
        AT = [true_answers[index] for index in sample_indices]

        batch_size = 5
        num_runs = np.int32(np.ceil(sample_size / batch_size))

        explanations = []
        ratings = []

        for i in tqdm(range(0, num_runs)):
            i_start = batch_size * i
            i_end = batch_size * (i + 1)

            questions_batch = Q[i_start:i_end]
            answers_batch = AP[i_start:i_end]

            verdicts = self.judge.answer(questions_batch, answers_batch)

            for (explanation, rating) in verdicts:
                explanations.append(explanation)
                ratings.append(rating)
        
        assert(len(explanations) == sample_size)
        assert(len(ratings) == sample_size)

        df_judges = pd.DataFrame({"qid" : QI, "Question" : Q, "True Answer" : AT, "Predicted Answer" : AP, "Rating" : ratings, "Explanations" : explanations})

        return df_judges

    def hallucination_detection_with_context(self, qids, questions, contexts, predicted_answers, true_answers):
        if not(self.loaded_lettuce):
            self.lettuce = LettuceDetect()
            self.loaded_lettuce = True
        if not(self.loaded_hallucinot):
            self.hallucinot = HalluciNot()
            self.loaded_hallucinot = True

        N = len(qids)

        assert(N == len(questions) and N == len(contexts) and N == len(predicted_answers))

        lettuce_results = []
        hallucinot_results = []

        for (question, context, answer) in zip (questions, contexts, predicted_answers):
            lettuce_results.append(self.lettuce.predict(question = question, context = context, answer = str(answer)))
            hallucinot_results.append(self.hallucinot.predict_with_context(question = question, context = context, answer = str(answer)))

        df = pd.DataFrame({"qid" : qids, "question" : questions, "True Answer" : true_answers, "Predicted Answer" : predicted_answers, "Lettuce" : lettuce_results, "Haluccinot" : hallucinot_results})

        return df
    
    def hallucination_detection_without_context(self, qids, questions, predicted_answers, true_answers):
        if not(self.loaded_hallucinot):
            self.hallucinot = HalluciNot()
            self.loaded_hallucinot = True

        N = len(qids)

        assert(N == len(questions) and N == len(predicted_answers))

        results = []

        for (question, answer) in zip (questions, predicted_answers):
            results.append(self.hallucinot.predict_without_context(question = question, answer = str(answer)))

        df = pd.DataFrame({"qid" : qids, "question" : questions, "True Answer" : true_answers, "Predicted Answer" : predicted_answers, "Haluccinot" : results})

        return df

    '''
        Model: An LLM model 
        Prompts: A list of prompts as strings
        true_answers: A list of the true answers as strings
        ids: A list of the qids.

        Returns: A pandas dataframe, with a row for every prompt and evaluation statistics.

        TOOD: Sort prompts by length to minimize padding and optimize performance.
    '''
    def evaluate(self, model, prompts, true_answers, ids, batch_size = 2):
        N = len(prompts)
        num_runs = np.int32(np.ceil(N / batch_size))
        time_start = time.time()
        cnt = 0

        J, R, P, F, E, EM, B, PA  = [], [], [], [], [], [], [], []
        T = []
        PE = []
        QID = []
        IDK = []

        time_total_inference = 0     


        for i in tqdm(range(0, num_runs)):
            i_start = batch_size * i
            i_end = batch_size * (i + 1)
            prompts_batch = prompts[i_start:i_end]

            # Logits batch has shape [Num_generated_tokens, Batch Size, Vocab Size]
            time_inference_start = time.time()
            predicted_answers_batch, logits_batch, num_tokens_in_answers = model.answer(prompts_batch)
            time_inference_end = time.time() 
            time_total_inference += time_inference_end - time_inference_start

            true_answers_batch = true_answers[i_start:i_end]
            ids_batch = ids[i_start:i_end]

            time_avg_in_batch = ((time_inference_end - time_inference_start) / 60.0) / batch_size

            for j, (pred_answer, true_answer, qid, answer_token_length) in enumerate(zip(predicted_answers_batch, true_answers_batch, ids_batch, num_tokens_in_answers)):
                
                # Index the i'th sample, to get logits of shape [Num_generated_tokens, Vocab Size]
                logits = logits_batch[:, j, :]
                # Extract only the logits that are for the answer token (i.e. exclude the logits for trailing padding token)
                logits = logits[0:answer_token_length, :]

                QID.append(qid)
                J.append(self.jaccard_similarity(true_answer, pred_answer))
                R.append(self.recall(true_answer, pred_answer))
                P.append(self.precision(true_answer, pred_answer))
                F.append(self.f1(true_answer, pred_answer))
                E.append(self.edit_distance(true_answer, pred_answer))
                EM.append(self.exact_match(true_answer, pred_answer))
                B.append(self.BERT_score(true_answer, pred_answer))
                PA.append(self.logits_to_perp_ans(logits))
                PE.append(pred_answer)
                T.append(time_avg_in_batch)

                if pred_answer.lower() == "idk":
                    IDK.append(1)
                else:
                    IDK.append(0)
                
                cnt += 1
        
        
        assert(cnt == N)
        # https://stackoverflow.com/questions/18837262/convert-python-dict-into-a-dataframe        
        # # https://www.geeksforgeeks.org/python/create-a-pandas-dataframe-from-lists/

        df = pd.DataFrame({"qid" : QID, "True Answer" : true_answers,"Predicted Answer" : PE, "Jaccard" : J, "Recall" : R, 
                           "Precision" : P, "F1" : F, "EditDist" : E, "EM" : EM, 
                           "BERT" : B, "Perp_ans" : PA, "IDK" : IDK,
                           "Time" : T})
        time_end = time.time() 

        # Total time in minutes.
        time_total = (time_end - time_start) / 60.0

        time_total_inference = time_total_inference / 60.0

        print("Time total to run evaluate = {}".format(time_total))
        print("Total Inference time = {}".format(time_total_inference))

        return df
    
    '''
        logits_sentence: Tensor of shape [Num_generated_tokens_excluding_padding, Vocab Size]

        Returns the perplexity of the answer assuming greedy decoding (i.e. extract max logits of all entries in Num_generated
        _tokens).
    '''
    def logits_to_perp_ans(self, logits_sentence):
        probs = []

        for i in range(logits_sentence.shape[0]):
            # Get logits for the individual token, so shape of [Num_generated_tokens]
            logits_token = logits_sentence[i, :]
            probs_token = torch.softmax(logits_token, dim = 0)
            max_prob = torch.max(probs_token)
            probs.append(float(max_prob))
        
        return self.perplexity(probs)
    
    '''
        token_ids_gold: Tensor of shape [Num_Tokens_Gold_ans] with the token ids of the golden truth answer

        Returns the perplexity of the gold answer of the model.
    '''
    def perp_gold(self, token_ids_gold):
        raise NotImplementedError("Perplexity of Gold truth has not been implemented yet")

    '''
        probs: A lit of probabilities, i.e. list of floats.
    '''
    # https://medium.com/@shubhamsd100/understanding-perplexity-in-language-models-a-detailed-exploration-2108b6ab85af
    # For numerical stability, first compute negative log likelihood, and then compute perplexity
    def perplexity(self, probs):
        N = len(probs)
        # np.log, is log with base e.
        NLL = -1.0/N * sum([np.log(p) for p in probs])
        perplexity = np.pow(np.e, NLL)

        return perplexity


    '''
        truth: A string with the gold truth answer
        pred: A string with the predicted answer    
    '''
    def jaccard_similarity(self, truth, pred):
        truth_tokens = word_tokenize(truth)
        pred_tokens = word_tokenize(pred)

        truth_set = set(truth_tokens)
        pred_set = set(pred_tokens)

        intersection = truth_set.intersection(pred_set)
        union = truth_set.union(pred_set)

        jaccard = len(intersection) / len(union)

        return jaccard
    
    '''
        truth: A string with the gold truth answer
        pred: A string with the predicted answer    
    '''
    def recall(self, truth, pred):
        truth_set = set(word_tokenize(truth))

        if len(truth_set) == 0:
            raise Exception("Truth Answer with zero tokens")

        pred_set = set(word_tokenize(pred))

        recalled_tokens = []
        for token in truth_set:
            if token in pred_set:
                recalled_tokens.append(token)

        if len(recalled_tokens) == 0:
            return 0
        else:
            return len(recalled_tokens) / len(truth_set)

    '''
        truth: A string with the gold truth answer
        pred: A string with the predicted answer    
    '''        
    def precision(self, truth, pred):
        pred_set = set(word_tokenize(pred))

        if len(pred_set) == 0:
            return 0
        
        truth_set = set(word_tokenize(truth))

        precision_tokens = []

        for token in pred_set:
            if token in truth_set:
                precision_tokens.append(token) 

        if len(precision_tokens) == 0:
            return 0
        
        return len(precision_tokens) / len(pred_set)
    
    

    '''
        truth: A string with the gold truth answer
        pred: A string with the predicted answer    
    ''' 
    def f1(self, truth, pred):
        recall = self.recall(truth, pred)
        precision = self.precision(truth, pred)

        if recall == 0 and precision == 0:
            return 0
        else:
            return 2.0 * (precision * recall) / (precision + recall)

    '''
        truth: A string with the gold truth answer
        pred: A string with the predicted answer    
    ''' 
    def exact_match(self, truth, pred):
        # Strip leading and trailing white space.
        return 1 if truth.strip(" ") == pred.strip(" ") else 0
    
    '''
        truth: A string with the gold truth answer
        pred: A string with the predicted answer    
    ''' 
    # https://www.geeksforgeeks.org/python/introduction-to-python-levenshtein-module/
    def edit_distance(self, truth, pred):
        return Levenshtein.distance(truth, pred)
    

    '''
        truth: A string with the gold truth answer
        pred: A string with the predicted answer    
    ''' 
    # https://huggingface.co/papers/1904.09675
    # https://medium.com/@abonia/bertscore-explained-in-5-minutes-0b98553bfb71
    # https://www.google.com/search?q=python+convert+to+lower+case&sca_esv=dc5fda282d29e3f3&sxsrf=ANbL-n7pE7f6H8in5V0YAabkEp3MOnCYEQ%3A1769517043592&source=hp&ei=8694aZ2xIpbwwPAP-KfA4AE&iflsig=AFdpzrgAAAAAaXi-A40lGRGoZxjhZJBmUhyktEECGbdG&ved=0ahUKEwjd9rjW3KuSAxUWOBAIHfgTEBwQ4dUDCCA&uact=5&oq=python+convert+to+lower+case&gs_lp=Egdnd3Mtd2l6IhxweXRob24gY29udmVydCB0byBsb3dlciBjYXNlMgcQABiABBgNMgYQABgWGB4yCBAAGBYYChgeMgYQABgWGB4yBhAAGBYYHjIGEAAYFhgeMggQABgWGAoYHjIIEAAYFhgKGB4yBhAAGBYYHjIGEAAYFhgeSNMYUABY-hdwAHgAkAEAmAFnoAHXDqoBBDI3LjG4AQPIAQD4AQGYAhygAvsPwgILEAAYgAQYsQMYgwHCAg4QABiABBixAxiDARiKBcICCBAAGIAEGLEDwgIFEAAYgATCAg4QLhiABBixAxiDARiKBcICCBAuGIAEGLEDwgIFEC4YgATCAgsQLhiABBixAxiDAcICBxAAGIAEGBPCAggQABgTGBYYHsICChAAGBMYFhgKGB6YAwCSBwQyNy4xoAePwAGyBwQyNy4xuAf7D8IHBjItMjcuMcgHeIAIAA&sclient=gws-wiz
    def BERT_score(self, truth, pred):

        if len(pred) == 0:
            return 0
        
        truth = truth.lower()
        pred = pred.lower()

        try:
            P, R, F1 = self.BERT_scorer.score([pred], [truth])
        except:
            return -1

        return float(F1[0])
    
    '''
        truth: A string with the gold truth answer
        pred: A string with the predicted answer 

        Computes chrf++
    '''
    # https://lightning.ai/docs/torchmetrics/stable/text/chrf_score.html
    def chrfPP(self, truth, pred):
        score = self.chrf_scorer([pred.strip()], [truth.strip()])
        return float(score)
        
