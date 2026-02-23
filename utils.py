"""
utils.py
--------
Shared utilities used across the pipeline.
Centralising the tokenizer here ensures BM25 index building
and BM25 querying ALWAYS use identical text processing.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)

_stop_words = set(stopwords.words("english"))
_stemmer    = PorterStemmer()


def tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25.

    Steps:
      1. Lowercase
      2. Remove punctuation / non-alphanumeric characters
      3. Split on whitespace
      4. Remove stopwords and very short tokens (len <= 2)
      5. Stem with Porter Stemmer

    IMPORTANT: Both the index builder (03_build_index.py) and the
    retriever (retrieve.py) import this function so that query tokens
    and document tokens are always processed identically.
    """
    text   = text.lower()
    text   = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in _stop_words and len(t) > 2]
    tokens = [_stemmer.stem(t) for t in tokens]
    return tokens
