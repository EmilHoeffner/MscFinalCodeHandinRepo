
import sys

from Evaluater import Evaluater

def main():
    evaluator = Evaluater()

    score1 = evaluator.BERT_score("yes", "no")
    score2 = evaluator.BERT_score("no", "yes")


    print(score1)
    print(score2)




if __name__ == "__main__":
    main()