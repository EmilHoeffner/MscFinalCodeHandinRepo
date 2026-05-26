# This file go thorugh the process of running the code to replicate all results and plots of the main experiment. 


# Getting results of the preliminary Experiment. 

Simply run the "LLMTest.ipynb" notebook in the "src/PrelimExperiment" folder. NOTE, that before doing this, you will need
to provide your own API keys from Huggingface and Gemini in the "keymanagement.py" file. 

# Getting the BM25 plot in section 2.4.3 of the report:

Navigate to "IllustrativePlots". Simply run the command:

python BM25K1Plot.py

Then the plot should pop up on your screen.

# Getting all the results of the tuning and main experiment takes some time. The "Storage - Results" folder has the results i ran and used in the report. By running all below commands, these should be replicated to the "Storage - Clean" folder.

# Storing datafiles and getting all retrieval results:

## Saving data files and retrieval index for the different datasplits. The datafiles are saved in "Storage - Clean/DataFiles/DATASETNAME". The index is stored in "Storage - Clean/Indexfiles/DATASETNAME". 

Navigate to "src/Retrieval" and run the following commands:

python index.py Squad_Tune

python index.py WikiMultiHop_Tune

python index.py Squad_Test

python index.py WikiMultiHop_Test

## Saving RetrievalResults for tuning splits. The pytterier retrievalfiles for each combination of b and k1 are stored in "Storage - Clean/RetrievalFiles/DATASETNAME/Modelfiles". The summary scores of the retrievers are stored in "Storage - Clean/RetrievalFiles/DATASETNAME/EvaluationFile". 

python retrieve_tune.py Squad_Tune

python retrieve_tune.py WikiMultiHop_Tune

python evaluate.py Squad_Tune

python evaluate.py WikiMultiHop_Tune

## Tuning plots of retrieval results. To get the tuning plots of the retrieval (i.e. plots that shows how e.g. MRR depends on b and k1). The results are saved in "Storage - Clean/TuningPlots". Run the following command:

python retrieve_plots.py

## Saving RetrievalResults for the test split. The retrieval files (i.e. pytterierfile with retreivalresults using the tuned BM25 for the test set) is going to be located in "Storage - Clean/RetrievalFiles/DATASETNAME/ModelFiles". The summary results are going to be located in "Storage - Clean/RetrievalFiles/DATASETNAME/EvaluationFile". Run the below commands:

python retrieve_test.py Squad_Test

python retrieve_test.py WikiMultiHop_Test

python evaluate.py Squad_Test 

python evaluate.py WikiMultiHop_Test

## Confidence intervals of retrieval summary statistics on test set. The confidence intervals are printed in the terminal by running the below commands:

python evaluate_confidence.py Squad_Test

python evaluate_confidence.py WikiMultiHop_Test

## Results of the retrieval error analysis on the test sets. The results used in the error analysis will be stored in "Storage - Clean/RetrievalFiles/DATASETNAME/ErrorAnalysis" by running below commands:

python errorAnalysis.py Squad_Test

python errorAnalysis.py WikiMultiHop_Test

# Dataset statistics from section 5. To get the Dataset statistics do the following:

Navigate to "src/Datasets"

Run the command:

python DatasetStatistics.py DATASETNAME  (e.g. replace DATASETNAME with Squad_Test)

Plots will then be stored in "Storage/DatasetStatistics/DATASETNAME


# Follow the instructions in "guide.txt" for runing inference with the models on the tuning and test sets. This gets results for all samples for M_question, M_RAG and M_ceil for the data splits. Please read this from start to end before running code in collab. It takes a lot of time to run all the code. If you dont want to do that, the results and runned versions of the notebooks are stored for all the models in the "CollabResults" folder. 

## After getting the collab results, to replicate results on all data splits in the report, please do the following first:

* Put csv files from "CollabResults/Squad_Test" in "Storage/EvaluationFiles/Squad_Test"

* Put csv files from "CollabResults/WikiMultiHop_Test" in "Storage/EvaluationFiles/WikiMultiHop_Test"

* Put csv files from "CollabResults/Squad_Tune_Zero" in "Storage/EvaluationFiles/Squad_Tune"

* Put csv files from "CollabResults/Squad_Tune_Few" in "Storage/EvaluationFiles/Squad_Tune_Few"

* Put csv files from "CollabResults/WikiMultiHop_Tune" in "Storage/EvaluationFiles/WikiMultiHop_Tune"

* Put csv files from "CollabResults/WikiMultiHop_Tune_Few" in "Storage/EvaluationFiles/WikiMultiHop_Tune_Few"

### Getting summary statistics and confidence intervals of summary statistics for the end-to-end experiment:

Navigate to "src/SummaryAnalysis". Run the following command:

python Analysis.py DATASETNAME

where DATASETNAME is e.g. Squad_Tune_Few or Squad_Test etc.

This will save file "SummaryAnalysis.csv" with the summary statistics and "Bootstrap_Question.csv", "Bootstrap_RAG.csv" and "Bootstrap_Ceil.csv" with the bootstrapped 95% confidence intervals of the models M_question, M_RAG and M_ceil. It is saved in the folder "Storage - Clean/SummaryAnalysis/DATASETNAME"

### Getting Results of the Attention Analysis:

This was only done on the test splits, as it was part of the main exploratory experiment and not the tuning experiment.

Navigate to "src/Evaluation"

For Squad, run:

python attentionAnalysis.py Squad_Test Zero

This will show the plots in the report and in the terminal, 10 good and hard samples of SQUAD are listed with the 
information listed in the table in the report.

For Wiki, to get the results from a random sample over all question types, run:

python attentionAnalysis.py WikiMultiHop_Test Zero

For Wiki, to get results over individual question types, run:

python attentionAnalysis.py WikiMultiHop_Test Zero comparison

python attentionAnalysis.py WikiMultiHop_Test Zero bridge_comparison

python attentionAnalysis.py WikiMultiHop_Test Zero compositional

python attentionAnalysis.py WikiMultiHop_Test Zero inference


### Getting Results of the Correlation Analysis:

This was only done on the test sets, as it was part of the main exploratory experiment and not the tuning experiment.

Navigate to "src/CorrelationAnalysis".

Below shows how to get results for "Squad_Test". Simply replace this by e.g.
"WikiMultiHop_Test" to get results on the WikiMultiHop test split. 

Run below command to get all correlation plots and correlatin coefficients between response/response, feature/feature and feature/response variables. This needs to be run 3 times, depending on whether you use F1, BERT or (F1 + BERT) / 2 as the response variable for feature/response correlation plots.

Running the first command takes quite some time, as it on the first run stores computed information such as 
readability measures of each context in csv files called "cache_train.csv" and "cache_test.csv".

Run the following commands:

python AnalyseCorrelation.py Squad_Test F1

python AnalyseCorrelation.py Squad_Test BERT

python AnalyseCorrelation.py Squad_Test Add (Using (F1 + BERT) / 2 as response variable.)

The results will be stored in the folder "Storage - Clean/CorrelationAnalysis/DATASETNAME".

To get the PCA and t-SNE projections, please run:

python VisualisationAnalysis.py Squad_Test F1

The results will also be stored in the folder "Storage - Clean/CorrelationAnalysis/DATASETNAME".

### Getting Results of the RegressionAnalysis

Navigate to "src/CorrelationAnalysis".

Run the commands:

python RegressionAnalysis.py Squad_Test

python RegressionAnalysis.py WikiMultiHop_Test

The numerical results in the report will be presented in the console. The plots of the report should pop up. 


### Getting Results of the Robustness Analysis.

Navigate to "src/RobustnessAnalysis"

Run the command:

python Analysis.py Squad_Test Zero Test

For WikiMultiHop_Test, simply replace Squad_Test with WikiMultiHop_Test. 
This will print results of the Robustness to irrelevant context analysis to the terminal.
You need to run the program twice, to get results of the counter factual analysis. The results will be stored in "RobustnessAnalysis/DATASETNAME/"


### Getting Results for comparing end-to-end results of tuning:

Navigate to "src/Evaluation"

Run the command:

python zero_few_comp.py Squad_Tune 

python zero_few_comp.py WikiMultiHop_Tune 

Numerical values used in the report are printed to the terminal. The plots are saved in the 
"Storage/TuningPlots" folder.

### Getting Results of the manual annotation:

Navigate to "src/Evaluation"

To store the random samples that i looked at, run:

python annotation_sampler.py Squad_Test

python annotation_sampler.py WikiMultiHop_Test

This will save a file "annotation.csv" to the "Storage - Clean/EvaluationFiles/DATASETNAME" folder which contains the samples I annotated.
You can take the corresponding file from the "Storage - Results" folder and run the following command to get the summary results of the analysis:

python annotation_evaluater.py Squad_Test

python annotation_evaluater.py WikiMultiHop_Test

### Getting Results of the context analysis on SQUAD

Navigate to "src/Evaluation"

Run the command:

python contextAnalysis.py Squad_Test BERT

The results are saved in the "Storage - Clean/ContextAnalysis" folder.