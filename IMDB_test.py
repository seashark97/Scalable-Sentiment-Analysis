import os
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import *
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import chi2, SelectKBest
import glob
import time
from nltk.stem.snowball import EnglishStemmer
from nltk.stem.porter import *
def give_text(directory):
	'''
	text generator for memory saving
	'''
	for i in range(12500):
		temp = open(glob.glob(directory+str(i)+'_'+'*.txt')[0], 'r')
		text = temp.read()
		temp.close()
		yield text
stemmer = PorterStemmer()
mygen = give_text('/Users/abdul/Desktop/RSI/test_code/train/pos/')
reviews = []

def negate_gram(document):
	doclist = document.split()
	review = []
	negated = False
	for n in range(len(doclist)):
		if doclist[n] == 'not' and not negated and len(doclist)>n+1:
			review.append(doclist[n]+'_'+doclist[n+1])
			negated = True
		elif not negated:
			review.append(doclist[n])
		elif negated:
			negated = False
	return ' '.join(review)


for n in mygen:
	# n = n.decode("utf8")
	# stemmed = ''
	# for word in n.split():
	# 	stemmed +=  stemmer.stem(word) +' '
	reviews.append(negate_gram(n))
mygen = give_text('/Users/abdul/Desktop/RSI/test_code/train/neg/')
for n in mygen:
	# n = n.decode("utf8")
	# stemmed = ''
	# for word in n.split():
	# 	stemmed +=  stemmer.stem(word) +' '
	reviews.append(negate_gram(n))

mygen = give_text('/Users/abdul/Desktop/RSI/test_code/test/pos/')
test_data = []
for n in mygen:
	# n = n.decode("utf8")
	# stemmed = ''
	# for word in n.split():
	# 	stemmed +=  stemmer.stem(word) +' '
	test_data.append(negate_gram(n))
mygen = give_text('/Users/abdul/Desktop/RSI/test_code/test/neg/')
for n in mygen:
	# n = n.decode("utf8")
	# stemmed = ''
	# for word in n.split():
	# 	stemmed +=  stemmer.stem(word) +' '
	test_data.append(negate_gram(n))

sentiments = [0 for n in range(12500)] + [1 for n in range(12500)]

hv = HashingVectorizer(ngram_range=(1,3), binary=False, non_negative=True)
X = hv.transform(reviews)
# selector = SelectKBest(chi2, k=65000)
# selector.fit(X, sentiments)
# X_reduced = selector.transform(X)
# clf = BernoulliNB()
# clf.fit(X_reduced, sentiments)
# print accuracy_score(sentiments, clf.predict(selector.transform(hv.fit_transform(test_data))))
current_max = [0,0]
for n in range(0, 100000, 5000):
	selector = SelectKBest(chi2, k=n)
	selector.fit(X, sentiments)
	X_reduced = selector.transform(X)
	clf = MultinomialNB()
	clf.fit(X_reduced, sentiments)
	accuracy = accuracy_score(sentiments, clf.predict(selector.transform(hv.fit_transform(test_data))))
	if accuracy > current_max[0]:
		current_max[0]=accuracy
		current_max[1]=n
	print 'Accuracy: %f    Number of features: %d' % (accuracy, n)
print 'Best number of features: %d    Accuracy: %f' % (current_max[1], current_max[0])



# print accuracy_score(sentiments, predicted), precision_score(sentiments, predicted), recall_score(sentiments, predicted)