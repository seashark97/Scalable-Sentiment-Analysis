''' Takes in training data and makes two variables: 
data: set of all data. full_sentences: all full sentences
###########
'''
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
import csv
import nltk
from nltk import bigrams, trigrams
import re
import numpy
from random import shuffle

data = []
with open('/Users/abdul/Desktop/RSI/Kaggle/Data/train.tsv') as tsv:
	tsv = csv.reader(tsv, delimiter = '\t')
	for row in tsv:
		data.append(row)


data = data[1:]
full_sentences = []
appended = []
sentiments = []
for phrase in data:
	if phrase[1] not in appended:
		appended.append(phrase[1])
		full_sentences.append(phrase[2])
		if int(phrase[3]) > 2:
			sentiments.append(1)
		#elif int(phrase[3]) == 2:
		#	sentiments.append(1)
		else:
			sentiments.append(0)
# unigrams = [nltk.word_tokenize(full_sentences[n]) for n in range(len(full_sentences))]
# bigrams = [bigrams(unigrams[n]) for n in range(len(full_sentences))]
# trigrams = [trigrams(unigrams[n]) for n in range(len(full_sentences))]

# allgrams = [unigrams[n] + bigrams[n] + trigrams[n] for n in range(len(full_sentences))]
# result = []
# for element in range(len(allgrams)):
# 	result += allgrams[element]
# allgrams = result[:]
# for element in range(len(allgrams)):
# 	if type(allgrams[element]) == tuple:
# 		allgrams[element] = ' '.join(allgrams[element])
print len(full_sentences)
hv = HashingVectorizer(ngram_range=(1,1), binary=True, non_negative=True)
X = hv.fit_transform(full_sentences[:len(full_sentences)/2])
clf = MultinomialNB().fit(X, sentiments[:len(full_sentences)/2])
predicted = clf.predict(hv.transform(full_sentences[len(full_sentences)/2:]))
print accuracy_score(sentiments[len(full_sentences)/2:], predicted)