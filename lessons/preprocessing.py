import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text, drop_stopwords=False):
    tokens = word_tokenize(text.lower())
    if drop_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    return tokens


def stem_tokens(tokens):
    return [STEMMER.stem(token) for token in tokens]


def lemmatize_tokens(tokens):
    return [LEMMATIZER.lemmatize(token) for token in tokens]


def sentence_split(text):
    return sent_tokenize(text)
