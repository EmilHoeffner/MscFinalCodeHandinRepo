from Evaluater import Evaluater
def main():
    evaluater = Evaluater(load_scorers = True)

    s = evaluater.chrfPP(truth = "no", pred = "no")

    print(s)

    s = evaluater.chrfPP(truth = "three", pred = "three")

    print(s)

    s = evaluater.chrfPP(truth = "three three", pred = "three three")

    print(s)

    s = evaluater.chrfPP(truth = "The sun was", pred = "The sun was")

    print(s)

    s = evaluater.chrfPP(truth = "The sun was shining", pred = "The sun was shining")

    print(s)

    print("\n")

    evaluater = Evaluater(load_scorers = True, character_overlap = 2, n_gram_overlap = 1)

    s = evaluater.chrfPP(truth = "no", pred = "no")

    print(s)

    s = evaluater.chrfPP(truth = "three", pred = "three")

    print(s)

    s = evaluater.chrfPP(truth = "three three", pred = "three three")

    print(s)

    s = evaluater.chrfPP(truth = "The sun was", pred = "The sun was")

    print(s)

    s = evaluater.chrfPP(truth = "The sun was shining", pred = "The sun was shining")

    print(s)

if __name__ == "__main__":
    main()