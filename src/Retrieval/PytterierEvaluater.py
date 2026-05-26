# https://pyterrier.readthedocs.io/en/latest/experiments.html

import pyterrier as pt

import sys 
import numpy as np

sys.path.append("..")

from SummaryAnalysis.SummaryAnalyser import SummaryAnalyser

class PytterierEvaluater:

    def __init__(self):
        if not pt.started():
            pt.init() 

    def evaluate(self, dfs, names, queries, qrels):
        # Convert all to string, since pytterier seems to expect it.
        queries["qid"] = [str(qid) for qid in list(queries["qid"])]
        queries["query"] = [str(query) for query in list(queries["query"])]

        qrels["qid"] = [str(qid) for qid in list(qrels["qid"])]
        qrels["docno"] = [str(docno) for docno in list(qrels["docno"])]
        
        return pt.Experiment(retr_systems = dfs, names = names, 
                             topics = queries, qrels = qrels,
                             eval_metrics = ["recall_1", "recall_5", "recall_10", "recip_rank"],
                             round = 8)
    
    def evaluate_confidence(self, dfs, names, queries, qrels, data_iden):
        # Convert all to string, since pytterier seems to expect it.
        queries["qid"] = [str(qid) for qid in list(queries["qid"])]
        queries["query"] = [str(query) for query in list(queries["query"])]

        qrels["qid"] = [str(qid) for qid in list(qrels["qid"])]
        qrels["docno"] = [str(docno) for docno in list(qrels["docno"])]

        assert(len(names) == 1)

        metrics = ["recall_1", "recall_5", "recall_10", "recip_rank"]

        df = pt.Experiment(retr_systems = dfs, names = names, 
                             topics = queries, qrels = qrels,
                             eval_metrics = metrics,
                             round = 4,
                             perquery = True)
        
        summary_analyser = SummaryAnalyser(data_iden)
        
        for i,metric in enumerate(metrics):
            df_metric = df[df["measure"] == metric]
            values = list(df_metric["value"])
            CI = summary_analyser.bootstrap(x = values, bootstrap_stats_function = np.mean, alpha = 0.05)

            print(f"95% CI for {metric} = {[{CI[0]}, {CI[1]}]}")


        


