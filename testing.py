from corpus_test import Corpus
from ngram import BasicNgram

c = Corpus("chomskycorpus.txt")
ngram = BasicNgram(4, c.tokens)
