This repo contains the code and datafiles used for this thesis. Below lists a description of the different files/folders:

* CollabResults: Results from running the code in "src/Collab" in Google Colab. 

* src: Folder containing the code used for this thesis

* Storage - Results: Folder containing results stored by running the code. Note, that some results/plots of the thesis are not here, as some of them were obtained by printing in the terminal and manually saving "pop-up" plots from matplotlib.show. The folder contains several sub folders, which each have individual subfolders for each split of the data "e.g. Squad_Test for the test split of SQUAD". A list of them:

-- ContextAnalysis: Contains results of the marginal context influence on QA performance on SQUAD (section 8.4.5 in the report)

-- CorrelationAnalysis: Contains results of the Correlation Analysis (section 8.4 in the report)

-- DataFiles: Contains datafiles of the sampled splits. Each sample has a qid associated with it. The datafiles are "queries.csv" with the questions, "docs.csv" with the contexts, "qrels.csv" denoting the ground truth document associated with each question, "answers.csv" with an answer for every question and "collab.csv" which is just used for when code is run in Google Colab.

-- DatasetStatistics: Contains DatasetStatistics plots (for section 5 in the report)

-- EvaluationFiles: Contains results of the M_question, M_RAG and M_ceil model for the runs over the tuning and test splits. 

-- IndexFiles: Contains the Indexes of the data splits

-- RetrievalFiles: Contains results of the retrievers. The folder "ModelFiles" are the pytterier files of the retrievers, "EvaluationFile" lists the summary performance measures of the retrievers and "ErrorAnalysis" is a folder with the plots of the error analysis of retrieval (section 8.1.3 in the report)

-- RobustnessAnalysis: Contains results of the RobustnessAnalysis (this is both robustness to irrelevant and wrong/counterfactual context)

-- SummaryAnalysis: Contains SummaryStatistics of M_question, M_RAG and M_ceil. Also has confidence intervals computed by bootstrapping. 

-- TuningPlots: Plots used for the tuning experiments (Section 7 of the report.)

* Storage - Clean: Identical to "Storage - Results" except it does not contain the results. If instructions in "RunCode.md" is followed, it should store the same results as "Storage - Results". Note that empty folders contain ".gitkeep" files. This is simply to allow pushing empty folders to github

* versions.md: The versions of libaries used for running the code. 