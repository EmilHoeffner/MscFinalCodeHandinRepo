import pyterrier as pt
from FileHandler import FileHandler

# Seems like when pytterier scores queries with documents of equal score, it uses the highest docnumber as the firstly fetched.

def main():
    pt.init() 
    file_handler = FileHandler("Storage - Correct")
    data_iden = "WikiMultiHop_Test"

    b = 0.9
    k1 = 1.0

    qrels = file_handler.load_df(dir = "DataFiles", dataset_name = data_iden, filename = "qrels.csv")
    queries = file_handler.load_df(dir = "DataFiles", dataset_name = data_iden, filename = "queries.csv")
    results_new = file_handler.load_df(dir = "RetrievalFiles", dataset_name = data_iden, filename = "ModelFiles/RetrievalResults_b={}_k1={}.csv".format(str(b), str(k1)))

    file_handler = FileHandler("Storage - Old")
    results_old = file_handler.load_df(dir = "RetrievalFiles", dataset_name = data_iden, filename = "ModelFiles/RetrievalResults_b={}_k1={}.csv".format(str(b), str(k1)))
    

    queries["qid"] = [str(qid) for qid in list(queries["qid"])]
    queries["query"] = [str(query) for query in list(queries["query"])]

    qrels["qid"] = [str(qid) for qid in list(qrels["qid"])]
    qrels["docno"] = [str(docno) for docno in list(qrels["docno"])]


    metrics = ["recall_1", "recall_5", "recall_10", "recip_rank"]

    df_new = pt.Experiment(retr_systems = [results_new], names = ["Res"], 
                            topics = queries, qrels = qrels,
                            eval_metrics = metrics,
                            round = 4,
                            perquery = True,
                            precompute_prefix = False)
    
    df_old = pt.Experiment(retr_systems = [results_old], names = ["Res"], 
                            topics = queries, qrels = qrels,
                            eval_metrics = metrics,
                            round = 4,
                            perquery = True,
                            precompute_prefix = False)
    
    results_old_test = results_old.copy()

    assert(results_old.equals(results_old_test))
    assert(not(results_old.equals(results_new)))
    assert(df_old.equals(df_new))
    
    
    #qids_to_check = [923, 924, 1846, 1847, 1996, 1997, 2256]
    qids_to_check = [578, 579]
    qids_to_check = [str(qid) for qid in qids_to_check]
    
    df_metric = df_new[df_new["measure"] == "recall_1"]
    df_metric = df_metric[df_metric["qid"].isin(qids_to_check)]

    print(df_metric)

    R = results_new[results_new["qid"].isin(qids_to_check)]
    R = R[R["rank"] == 0.0]

    print(R)

    R = results_old[results_old["qid"].isin(qids_to_check)]
    R = R[R["rank"] == 0.0]

    print(R)


    retrieve_results = []
    qids = []
    docnos = []

    for i, row in results_new.iterrows():
        qid = row["qid"]
        docno = row["docno"]

        if qid in qids:
            continue

        if int(qid) == int(docno):
            retrieve_results.append(1.0)
            qids.append(qid)
            docnos.append(docno)
        else:
            retrieve_results.append(0.0)
            qids.append(qid)
            docnos.append(docno)
    

    df_new = df_new[df_new["measure"] == "recall_1"]
    df_new["qid"] = [int(qid) for qid in list(df_new["qid"])]
    df_new = df_new.sort_values("qid")
    pytterier_results = list(df_new["value"])
    pytterier_qids = list(df_new["qid"])

    #print(pytterier_qids)
    #print(qids)

    assert(len(pytterier_results) == len(retrieve_results))

    #print(sum(retrieve_results))
    #print(len(retrieve_results))

    ratio = len([res for i, res in enumerate(retrieve_results) if res == pytterier_results[i]]) / len(pytterier_results)

    print(ratio)
    
    

if __name__ == "__main__":
    main()