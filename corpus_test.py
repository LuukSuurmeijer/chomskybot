# Computational Linguistics, Assignment 1: Ngrams
# Universitaet des Saarlandes
# 04-11-2019

import re
import random
import nltk
import nltk.data
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
		#self.rawdata = re.sub("(\n)+", " ", self.rawdata)
		self.rawdata = re.sub(r"[^\w\s\,\.]","", self.rawdata)
		tokens = nltk.word_tokenize(self.rawdata)
		tokens = [re.sub('[0-9]', '', token) for token in tokens]
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

		

	def sentgen(self, ngram, w=0, seed=''):
		"""
		Generate a sentence (with nltk.generate) with minimum w amount of words based on an ngram of the corpus.
		Sentence is generated until a period is predicted to retain fluency.
	
		"""
		#if theres no minimum length, the minimum is the size of 1 context
		#if w == 0:
		#	w = ngram._n
	
		#if the user inputs a seed
		if seed:
			sent = seed.split(' ')
			context = tuple(sent)
		else:
			context = random.choice(ngram.contexts())
			sent = list(context)

		#user can input minimum length of a sentence
		#if a length w is inputted, function will generate until theres a period and it has already generated more words than the minimum
		#otherwise w is the length of 1 context, so it will always have generated more than the minimum length

		nextword = ngram[context].generate() #generate first new word
		while ((nextword != '.') or (len(sent) <=w)): #both of these conditions must evaluate false to stop iterating
			sent.append(nextword) 
		#append the newly generated word to the context (for the next word), shake off the first word of the previous context.
			context = tuple([x for x in context[1:]] + [nextword])
			nextword = ngram[context].generate() #generate all the other words
		sent.append(nextword) #append last word (=period)
		return ' '.join(sent)
	
	def quotify(self, sentence): #format a string nicely with capitals and punctuation
		sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') 
		sentences = sent_tokenizer.tokenize(sentence)
		sentences = [sent.capitalize() for sent in sentences]
		new = ' '.join(sentences)
		final = re.sub(r'\ (?=,|\.)', '', new)
		return final

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


def main():
	c = Corpus("chomskycorpus.txt")
	ngram = BasicNgram(4, c.tokens)

	s = c.sentgen(ngram, 20)
	print(s)
	print(c.quotify(s))
if __name__ == '__main__':
	main()

