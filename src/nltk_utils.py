import nltk
# nltk.download("punkt")
import numpy as np
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


def tokenize(sentence: str) -> list[str]:
    """Tokenize a sentence into a list of words

    Args:
        sentence (str): Raw input sentence.

    Returns:
        (list[str]): tokenized sentence.
    """
    return nltk.word_tokenize(sentence)


def stem(word: str) -> str:
    """Stems word using PorterStemmer

    Args:
        word (str): non-stemmed word.

    Returns:
        (str): stemmed word.
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence: list[str], all_words: list[str]) -> np.array:
    """Create a bag of words vector for a given sentence.

    Args:
        tokenized_sentence (list[str]): tokenized sentence.
        all_words (list[str]): list of all words in the dataset.

    Returns:
        (np.array): bag of words vector.
    """
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag
