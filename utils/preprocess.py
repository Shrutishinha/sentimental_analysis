"""
=========================================================
NLP Preprocessing Utilities
=========================================================
Functions Included

✔ Lowercase Conversion
✔ Remove HTML Tags
✔ Remove URLs
✔ Remove Email IDs
✔ Remove Numbers
✔ Remove Punctuation
✔ Remove Extra Spaces
✔ Tokenization
✔ Stopword Removal
✔ Stemming
✔ Lemmatization
✔ POS Tagging
✔ Named Entity Recognition
=========================================================
"""

import re
import string

import nltk
import spacy

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# -------------------------------------------------------
# Download Required Resources
# -------------------------------------------------------

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")
try:
    nltk.download("averaged_perceptron_tagger_eng")
except:
    pass

# -------------------------------------------------------
# Load spaCy Model
# -------------------------------------------------------

try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None

# -------------------------------------------------------
# Initialize Objects
# -------------------------------------------------------

STOPWORDS = set(stopwords.words("english"))

stemmer = PorterStemmer()

lemmatizer = WordNetLemmatizer()

# -------------------------------------------------------
# Basic Cleaning Functions
# -------------------------------------------------------

def lowercase(text):
    return text.lower()


def remove_html(text):
    return re.sub(r"<.*?>", "", text)


def remove_urls(text):
    return re.sub(r"http\S+|www\S+", "", text)


def remove_emails(text):
    return re.sub(r"\S+@\S+", "", text)


def remove_numbers(text):
    return re.sub(r"\d+", "", text)


def remove_punctuation(text):
    return text.translate(
        str.maketrans("", "", string.punctuation)
    )


def remove_extra_spaces(text):
    return re.sub(r"\s+", " ", text).strip()


# -------------------------------------------------------
# Tokenization
# -------------------------------------------------------

def tokenize(text):
    return word_tokenize(text)


# -------------------------------------------------------
# Stopword Removal
# -------------------------------------------------------

def remove_stopwords(tokens):

    return [
        word
        for word in tokens
        if word.lower() not in STOPWORDS
    ]


# -------------------------------------------------------
# Stemming
# -------------------------------------------------------

def stem_words(tokens):

    return [
        stemmer.stem(word)
        for word in tokens
    ]


# -------------------------------------------------------
# Lemmatization
# -------------------------------------------------------

def lemmatize_words(tokens):

    return [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]


# -------------------------------------------------------
# POS Tagging
# -------------------------------------------------------

def pos_tagging(tokens):

    return nltk.pos_tag(tokens)


# -------------------------------------------------------
# Named Entity Recognition
# -------------------------------------------------------

def named_entities(text):

    if nlp is None:
        return []

    doc = nlp(text)

    entities = []

    for ent in doc.ents:

        entities.append(
            (
                ent.text,
                ent.label_
            )
        )

    return entities


# -------------------------------------------------------
# Complete Cleaning Pipeline
# -------------------------------------------------------

def clean_text(text):

    text = lowercase(text)

    text = remove_html(text)

    text = remove_urls(text)

    text = remove_emails(text)

    text = remove_numbers(text)

    text = remove_punctuation(text)

    text = remove_extra_spaces(text)

    return text


# -------------------------------------------------------
# Full NLP Pipeline
# -------------------------------------------------------

def preprocess_text(text):

    cleaned = clean_text(text)

    tokens = tokenize(cleaned)

    tokens_no_stop = remove_stopwords(tokens)

    stemmed = stem_words(tokens_no_stop)

    lemmatized = lemmatize_words(tokens_no_stop)

    return {
        "original": text,
        "cleaned": cleaned,
        "tokens": tokens,
        "without_stopwords": tokens_no_stop,
        "stemmed": stemmed,
        "lemmatized": lemmatized,
        "processed_text": " ".join(lemmatized)
    }


# -------------------------------------------------------
# Sentiment Ready Text
# -------------------------------------------------------

def get_processed_text(text):

    result = preprocess_text(text)

    return result["processed_text"]


# -------------------------------------------------------
# Vocabulary Statistics
# -------------------------------------------------------

def vocabulary_size(text):

    tokens = tokenize(clean_text(text))

    return len(set(tokens))


def total_words(text):

    tokens = tokenize(clean_text(text))

    return len(tokens)


def character_count(text):

    return len(text)


def sentence_count(text):

    sentences = re.split(r"[.!?]", text)

    sentences = [
        s for s in sentences
        if s.strip()
    ]

    return len(sentences)


# -------------------------------------------------------
# Complete NLP Analysis
# -------------------------------------------------------

def analyze_text(text):

    result = preprocess_text(text)

    result["characters"] = character_count(text)

    result["words"] = total_words(text)

    result["sentences"] = sentence_count(text)

    result["vocabulary"] = vocabulary_size(text)

    result["pos_tags"] = pos_tagging(
        result["tokens"]
    )

    result["entities"] = named_entities(text)

    return result


# -------------------------------------------------------
# Testing
# -------------------------------------------------------

if __name__ == "__main__":

    sample = """
    <h1>I absolutely loved this movie!</h1>
    The acting was amazing and the story was wonderful.
    Visit https://example.com
    Contact me at abc@gmail.com
    """

    analysis = analyze_text(sample)

    print("=" * 60)
    print("Original")
    print("=" * 60)
    print(analysis["original"])

    print("\nCleaned")
    print(analysis["cleaned"])

    print("\nTokens")
    print(analysis["tokens"])

    print("\nWithout Stopwords")
    print(analysis["without_stopwords"])

    print("\nStemmed")
    print(analysis["stemmed"])

    print("\nLemmatized")
    print(analysis["lemmatized"])

    print("\nPOS Tags")
    print(analysis["pos_tags"])

    print("\nNamed Entities")
    print(analysis["entities"])

    print("\nStatistics")
    print("Characters :", analysis["characters"])
    print("Words :", analysis["words"])
    print("Sentences :", analysis["sentences"])
    print("Vocabulary :", analysis["vocabulary"])