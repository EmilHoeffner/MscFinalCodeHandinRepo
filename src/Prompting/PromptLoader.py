
import sys 

# Gets called from inside Evaluation and RobustnessAnalysis directory, so append "../Prompting" to path
sys.path.append("../Prompting")

from FewShotPrompter import FewShotPrompter
from ZeroShotPrompter import ZeroShotPrompter
from WeightPrompter import WeightPrompter

class PromptLoader:

    def __init__(self):
        pass 

    def load_prompter(self, iden):
        if iden == "Few":
            return FewShotPrompter()
        elif iden == "Zero":
            return ZeroShotPrompter()
        elif iden == "Weight":
            return WeightPrompter()
        else:
            raise Exception("Unknown Prompter Identifier")

