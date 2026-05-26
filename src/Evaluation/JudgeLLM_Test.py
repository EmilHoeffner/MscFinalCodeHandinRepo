import sys

# https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder
sys.path.append("..")

from LLM.JudgeLLM import JudgeLLM

import time

def main():
    judge = JudgeLLM()

    question = "Generally, how many legs does a dog have?"

    ans1 = "A dog generally has 4 legs"
    ans2 = "4 legs"
    ans3 = "A dog can have 3 legs"
    ans4 = "2 legs and 2 arms"
    ans5 = "yes"
    ans6 = "A dog generally has 4 legs, but can have fewer if it loses some"

    time_start = time.time()

    answers = [ans1, ans2, ans3, ans4, ans5, ans6]
    questions = [question for answer in answers]

    verdicts = judge.answer(questions, answers)

    for (explanation, rating), ans in zip(verdicts, answers):
        print(f"\nJudgement of answer: {ans}:\nExplanation: {explanation}\nRating: {rating}\n")

    time_end = time.time()

    print("\nTotal Time in Seconds: {}".format(time_end - time_start))

if __name__ == "__main__":
    main()