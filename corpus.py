# Computational Linguistics, Assignment 1: Ngrams
# Universitaet des Saarlandes
# 04-11-2019

import re
import random
import nltk
import numpy
from ngram import BasicNgram
import matplotlib
import matplotlib.pyplot

class Corpus:
	"""
	Class to :
	read a corpus,
	tokenize a corpus,
	generate frequency plots,
	generate a sentence based on an ngram of the corpus
	calculate PMI scores for word pairs in a corpus

	"""

	def __init__(self, filename, encoding="utf-8"):

		self._filename = filename 			#filename
		self._encoding = encoding 			#encoding
		self.rawdata = self._read() 		#just the raw data as a string, nothing done with it
		self.tokens = self._tokenize() 		#list of tokens
		self.frequency = self.freq() 		#dictionary of individual word frequencies
		self.tuples = self.freqtuples()		#dictionary of frequencies of word pairs

	def _read(self):
		"""
		Read the corpus file and return it as a string in rawdata
		"""
		with open(self._filename, "r", encoding=self._encoding) as f:
			rawdata = f.read()
		return rawdata

	def _tokenize(self):
		"""
		Return a list of tokens based on the raw data of a corpus.
		"""
		# do lower case, replace newlines and remove everything that inst a word (i keep comma's and dots for readability)
		self.rawdata = self.rawdata.lower()
		self.rawdata = re.sub("(\n)+", " ", self.rawdata)
		self.rawdata = re.sub(r"[^\w\s\,]","", self.rawdata)
		tokens = nltk.word_tokenize(self.rawdata)
		return tokens

	def freq(self):
		"""
		Count the occurences of all unique words in a corpus and return it as a dictionary
		"""
		frequency = {}
		for word in self.tokens:
			if word in frequency:
				frequency[word] += 1
			else:
				frequency[word] = 1
		return frequency
	
	def freqtuples(self):
		"""
		Count the occurences of all unique word pairs (w1w2) in a corpus and return it as a dictionary
		"""
		tuples = {}
		i = 0
		for i in range(len(self.tokens)-1):
			if (self.tokens[i], self.tokens[i+1]) in tuples:
				tuples[(self.tokens[i], self.tokens[i+1])] += 1
			else:
				tuples[(self.tokens[i], self.tokens[i+1])] = 1
		return tuples
			

	def getplots(self, log = False):
		"""
		Generate loglog/linear plots of the frequency of the words in a corpus. Used for problem 1
		"""
		if log:
			matplotlib.pyplot.loglog(sorted(self.frequency.values(), reverse = True))
			matplotlib.pyplot.title(f"Log-Log plot of f(words) in {self._filename}")
		else:
			matplotlib.pyplot.plot(sorted(self.frequency.values(), reverse = True))
			matplotlib.pyplot.title(f"Linear plot of f(words) in {self._filename}")

		matplotlib.pyplot.show()

	def sentgen(self, w, ngram, seed=''):
		"""
		Generate a sentence (with nltk.generate) with w amount of words based on an ngram of the corpus.
		
		Choose the first word by choosing randomly (or a seed), then generate a word based on the previous context.
		the new word becomes part of the context for the next word
		do until the string is w words long. Used for problem 2

		"""

		if seed:
			sent = seed.split(' ')
			context = tuple(sent)
		else:
			context = random.choice(ngram.contexts())
			sent = list(context)

		for k in range(w-ngram._n+1):
			nextword = ngram[context].generate()
			sent.append(nextword) 
		#append the newly generated word to the context (for the next word), shake off the first word of the previous context.
			context = tuple([x for x in context[1:]] + [nextword])
		return ' '.join(sent)
	
	def pmi(self, w1, w2):
		"""
		Calculate and return Pointwise Mutual Information (PMI) of a wordpair (consisting of (w1) and (w2)) according to the formula:
		
				C(w1w2) * N
		PMI =	-------------
				C(w1) * C(w2)

		Where N is the size of the corpus and C() is a function that returns the count (frequency) of the input.
		"""
		pmi = numpy.log((self.tuples[(w1, w2)]*len(self.tokens)) / 
						(self.frequency[w1] * self.frequency[w2]))
		return pmi

	def totalpmi(self):

		"""
		Calculate the PMI for every wordpair in a corpus and return it as a list of tuples:
		example: [((i, am), 10), ((am, a), 8)] etc. Used for problem 3.

		"""
		pmi_values = []
		for w1, w2 in self.tuples.keys():
			if (self.frequency[w1] or self.frequency[w2]) < 10:
				continue
			independence_value = self.pmi(w1, w2)
			pmi_values.append(((w1, w2), independence_value))
		return pmi_values


