# https://huggingface.co/blog/adaamko/lettucedetect

# https://huggingface.co/AimonLabs/hallucination-detection-model

from lettucedetect.models.inference import HallucinationDetector as Lettuce
from hdm2 import HallucinationDetectionModel

import torch
import numpy as np

class LettuceDetect:
    def __init__(self):
        self.detector = Lettuce(
            method="transformer", model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1", device_map = "cuda"
            )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def predict(self, question, context, answer):
        response = self.detector.predict(context = [context], answer = answer, question = question, output_format = "spans")
        return response


# https://huggingface.co/AimonLabs/hallucination-detection-model
class HalluciNot:
    def __init__(self):
        self.detector = HallucinationDetectionModel()

    def predict_with_context(self, question, context, answer):
        results = self.detector.apply(question, context, answer, debug = False)
        return self.format_results(results)
    
    def predict_without_context(self, question, answer):
        results = self.detector.apply(prompt = question, context = "", response = answer, debug = False)
        return self.format_results(results)
        

    def format_results(self, results):
        max_halu_value = 0
            
        # Print hallucinated sentences
        if results['candidate_sentences']:
            for sentence_result in results['ck_results']:
                if sentence_result['prediction'] == 1:  # 1 indicates hallucination
                    prob = sentence_result['hallucination_probability']

                    if prob > max_halu_value:
                        max_halu_value = float(prob)

        if max_halu_value == 0:
            s = "NOHALU"
        else:
            s = f"fHALU:{round(max_halu_value, ndigits = 4)}"

        return s
        
