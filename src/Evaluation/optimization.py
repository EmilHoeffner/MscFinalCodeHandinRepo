
import os 
import sys
from Evaluater import Evaluater

sys.path.append("..")

from Prompting.PromptLoader import PromptLoader
from Dataloading.Dataloader import DataLoader
from LLM.Flan import FlanT5
from Filehandling.FileHandler import FileHandler
from tqdm import tqdm
from LLM.Provence import Provence

# Tried the original prune method with threshold 0.25 which lead to slighty worse performance than the M_context model.
# Tried using chunk_then_prine with threshold 0.01, which led to worse performance than M_context model.
# Tried using weighting with top_proportion = 0.2, which led to worse performance than M_context model.

# Idea:
# Train a sequence labeller, to label every token as relevant or non-relevant for the question. Use ground truth passages from WikiMultihop as I in every token.


def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # https://www.nielit.gov.in/gorakhpur/sites/default/files/Gorakhpur/ALevel_1_Python_20May.pdf
    num_args = len(sys.argv)

    if num_args != 3:
        raise Exception("Program must be run with exactly two argument, denoting which dataset to evaluate over and which prompter to use")
    
    dataset_iden = sys.argv[1]
    prompter_iden = sys.argv[2]

    prompt_engineer = PromptLoader().load_prompter(prompter_iden)
    model = FlanT5()
    file_handler = FileHandler()

    dl = DataLoader()
    dataset = dl.load_data(dataset_iden)


    context_pruner = Provence()
    evaluater = Evaluater(load_scorers = True)

    qids = dataset.get_qids()[1900:2000]
    questions = dataset.get_questions()[1900:2000]
    contexts = dataset.get_contexts()[1900:2000]
    true_answers = dataset.get_answers()[1900:2000]

    # Prune the contexts:
    print("Pruning Contexts...")
    contexts = [context_pruner.weigh_sentences(questions[i], context) for i,context in enumerate(contexts)]

    prompts = prompt_engineer.construct_prompts_with_context(questions, contexts)

    results_df = evaluater.evaluate(model, prompts, true_answers, qids, batch_size = 1)

    file_handler.save_df(df = results_df, dir = "EvaluationFiles", dataset_name = dataset.get_name() , filename = "M_optim.csv", header = True)


if __name__ == "__main__":
    main()