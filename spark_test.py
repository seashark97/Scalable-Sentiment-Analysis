import os
from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.linalg import Vectors
from sklearn.feature_extraction.text import HashingVectorizer
from scipy.sparse import csc_matrix

conf = SparkConf().setAppName('sentimentAnalysis').setMaster('local')
sc = SparkContext(conf=conf)

def ngram_range(text):
	'''
	returns 1, 2, 3-grams without punctuation
	'''
	exclude = set('.,\'"\\/<>()~?!:;()][{}_@#')
	words = ''.join(ch for ch in text if ch not in exclude).lower().split()
	bigrams = [' '.join([words[n-1], words[n]]) for n in range(1,len(words))]
	trigrams = [' '.join([words[n-2], words[n-1], words[n]]) for n in range(2,len(words))]
	return words + bigrams + trigrams



amazon_dataset = sc.textFile('all.txt')



reviews = sc.parallelize(amazon_dataset.filter(lambda l: 'review/text:' in l).map(lambda l: l[13:]).take(200000))#.sample(False, 0.01, seed=42)
sentiments = sc.parallelize(amazon_dataset.filter(lambda l: 'review/score:' in l).map(lambda l: 1 if float(l[14:]) > 2.5 else 0).take(200000))#.sample(False, 0.01, seed=42)

indexed_reviews = reviews.zip(sc.parallelize([n for n in range(200000)]))
indexed_sentiments = sentiments.zip(sc.parallelize([n for n in range(200000)]))

train_reviews = indexed_reviews.sample(False, 0.5, 42)
train_sentiments = indexed_sentiments.sample(False, 0.5, 42)
# vocab = train_reviews.flatMap(ngram_range).distinct()

hv = HashingVectorizer(binary=True, ngram_range=(1,3))
hv_global = sc.broadcast(hv)
train_reviews.map(lambda r: hv_global.value.partial_fit(r[0]))
train_indices = train_reviews.map(lambda r: r[1])
train_indices = train_indices.collect()
test_reviews = indexed_reviews.filter(lambda review: review[1] not in train_indices)
test_sentiments = indexed_sentiments.filter(lambda sentiment: sentiment[1] not in train_indices)
# vocab_size = sc.broadcast(vocab.count())
# sorted_vocab = sorted(vocab.collect(), key=str.lower)
# vocab = sc.broadcast(zip(sorted_vocab, [n for n in range(vocab_size.value)]))

# def getIndex(word, mid=vocab_size.value/2, vocab=vocab.value):
# 	if word == vocab[mid]:
# 		return mid
# 	if mid == 1:
# 		return -1
# 	elif word < vocab[mid]:
# 		return getIndex(word,mid=mid/2, vocab=vocab[:mid])
# 	return getIndex(word,mid=mid+mid/2, vocab=vocab[mid:])
# vocab_dict = dict(vocab.zip(sc.parallelize([0 for n in range(vocab.count())])).collect())    #vocabulary dictionary. Format:  {'n-gram':0 for every n-gram in training set}
# vocab_dict.cache()
def vectorizer(text):
	'''
	takes in a dictionary of the vocabulary and makes a binary feature vector where the features are word occurence
	'''
	# i=0
	# nonzero = []
	# words = ngram_range(text)
	
	# for word in vocab:
	# 	if word in words:
	# 		nonzero.append((i, 1))
	# 	# else:
	# 	# 	nonzero.append((i, 0))
	# 	i+=1
	# return SparseVector(i, nonzero)
	# words = ngram_range(text)
	# copy = vocab_dict.copy()
	# for word in words:
	# 	if word in copy:
	# 		copy[word] = 1
	# vector = []
	# return copy.values()
	indices = hv_global.value.transform([text]).nonzero()
	return Vectors.sparse(2**20, indices[1], [1 for n in range(len(indices[0]))])
# train_text_vectors = train_text.map(vectorizer)
print '#########################################################'
print 'Checkpoint 1'
# training_vectors = train_sentiments.zip(train_text.map(lambda text: SparseVector(vocab_size, vocab.map(lambda ngram: i if ),[1 for n in range(vocab_size)]))#vectorizer)).collect() #.map(lambda labeled: LabeledPoint(labeled[0], labeled[1])).collect()
# def getIndexOnly(doc):
# 	indices = []
# 	for word in doc:
# 		if getIndex(word) != -1:
# 			indices.append(getIndex(word))
# 	return indices
# train_text_vectors = train_text.map(ngram_range).map(getIndexOnly)
# training_vectors = train_sentiments.zip(train_text_vectors.map(lambda doc: SparseVector(vocab_size.value, doc, [1 for n in range(len(doc))]))).map(LabeledPoint).collect()
train_sentiments_noindex = train_sentiments.map(lambda s: s[0])
train_reviews_noindex = train_reviews.map(lambda r: r[0])
review_vectors = train_reviews_noindex.map(vectorizer)
training_vectors = train_sentiments_noindex.zip(review_vectors).map(lambda g: LabeledPoint(g[0], g[1]))
print '#########################################################'
print 'Checkpoint 1.5'
clf = NaiveBayes.train(training_vectors)
print '#########################################################'
print 'Checkpoint 2'
test_vectors = test_reviews.map(lambda r: vectorizer(r[0]))
predictions = test_vectors.map(clf.predict)
print predictions.collect()
num_correct = float(predictions.zip(test_sentiments.map(lambda s: s[0])).filter(lambda x: x[0] == x[1]).count())
num_total = float(predictions.count())

print num_correct, num_total, num_correct/num_total