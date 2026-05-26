
from torchmetrics.text import BLEUScore


def main():
    bleu = BLEUScore()

    print(bleu(["no"], [["no"]]))
    print(bleu(["three"], [["three"]]))
    print(bleu(["three three"], [["three three"]]))
    print(bleu(["The sun was"], [["The sun was"]]))
    print(bleu(["The sun was shining"], [["The sun was shining"]]))


if __name__ == "__main__":
    main()