# https://pypi.org/project/ReadabilityCalculator/
from readcalc import readcalc

import re
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt

from wordcloud import WordCloud
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
# https://pypi.org/project/py-readability-metrics/
from readability import Readability

from collections import Counter
from nltk.corpus import stopwords

def lix(text):
    calc = readcalc.ReadCalc(text)
    return calc.get_lix_index()

def Automated_Readability(text):
    calc = readcalc.ReadCalc(text)
    return calc.get_ari_index()

def Coleman_liau(text):
    calc = readcalc.ReadCalc(text)
    return calc.get_coleman_liau_index()

def Flesh_Kincaid(text):
    calc = readcalc.ReadCalc(text)
    return calc.get_flesch_kincaid_grade_level()

def Gunning_Fox(text):
    calc = readcalc.ReadCalc(text)
    return calc.get_gunning_fog_index()

def Dale_Chall(text):
    calc = readcalc.ReadCalc(text)
    return calc.get_dale_chall_score()

'''
    texts: A list of strings
'''
# https://www.geeksforgeeks.org/python/generating-word-cloud-python/¨
# https://www.datacamp.com/tutorial/wordcloud-python
def word_cloud(texts, save_path, title, max_words = 10):
    cleaned_texts = [clean_text_for_word_cloud(text) for text in texts]
    # https://stackoverflow.com/questions/497765/string-joinlist-on-object-array-rather-than-string-array
    joined_text = ''.join([x for x in cleaned_texts])
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words = max_words, colormap='coolwarm', collocations = False).generate(joined_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

'''
    tokenizer: A function that returns a list of tokens of a string.
'''
# https://www.geeksforgeeks.org/python/counters-in-python-set-1/
# https://www.geeksforgeeks.org/nlp/removing-stop-words-nltk-python/
# https://www.w3schools.com/Python/matplotlib_bars.asp
def word_histogram(texts, save_path, title, tokenizer, max_words = 10):
    tokens = [tokenizer(text.lower()) for text in texts]
    tokens_total = []

    for lst in tokens:
        tokens_total += lst 

    stop_words = list(set(stopwords.words('english')))
    punctuations = [".", ",", "'", "(", ")", "'s", ":", ";", "''", "``"]
    tokens_total = [token for token in tokens_total if token not in (stop_words + punctuations)]
    
    counts = Counter(tokens_total)
    common = counts.most_common(max_words)

    top_tokens = [tok for (tok, freq) in common]
    top_frequencies = [freq for (tok, freq) in common]

    plt.figure(figsize=(10, 5))
    plt.bar(x = top_tokens, height = top_frequencies)
    plt.title(title)
    plt.xlabel("Token")
    plt.ylabel("Frequency")
    plt.savefig(save_path)
    plt.close()

# https://www.geeksforgeeks.org/python/generating-word-cloud-python/
def clean_text_for_word_cloud(text):
    # Remove punctuation
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Convert to lower
    text = text.lower()
    # Remove stopwords
    stopwords = set(STOPWORDS)
    text = ' '.join(word for word in text.split() if word not in stopwords)

    return text
