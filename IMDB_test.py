import os
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import *
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import chi2, SelectKBest
import glob
import time

def give_text(directory):
	for i in range(12500):
		temp = open(glob.glob(directory+str(i)+'_'+'*.txt')[0], 'r')
		text = temp.read()
		temp.close()
		yield text

mygen = give_text('/Users/abdul/Desktop/RSI/test_code/train/pos/')
reviews = []
for n in mygen:
	reviews.append(n)
mygen = give_text('/Users/abdul/Desktop/RSI/test_code/train/neg/')
for n in mygen:
	reviews.append(n)

mygen = give_text('/Users/abdul/Desktop/RSI/test_code/test/pos/')
test_data = []
for n in mygen:
	test_data.append(n)
mygen = give_text('/Users/abdul/Desktop/RSI/test_code/test/neg/')
for n in mygen:
	test_data.append(n)

sentiments = [0 for n in range(12500)] + [1 for n in range(12500)]

hv = HashingVectorizer(ngram_range=(1,1), binary=True)
X = hv.transform(reviews)
current_max = [0,0]
for n in range(0, 100000, 5000):
	selector = SelectKBest(chi2, k=n)
	selector.fit(X, sentiments)
	X_reduced = selector.transform(X)
	clf = BernoulliNB()
	clf.fit(X_reduced, sentiments)
	accuracy = accuracy_score(sentiments, clf.predict(selector.transform(hv.fit_transform(test_data))))
	if accuracy > current_max[0]:
		current_max[0]=accuracy
		current_max[1]=n
	print 'Accuracy: %f    Number of features: %d' % (accuracy, n)
print 'Best number of features: %d' % current_max[1]


# print accuracy_score(sentiments, predicted), precision_score(sentiments, predicted), recall_score(sentiments, predicted)