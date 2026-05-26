
from transformers import AutoTokenizer

def main():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl") 

    example = "This is a b sentence, hello."

    print(tokenizer.tokenize(example))



if __name__ == "__main__":
    main()