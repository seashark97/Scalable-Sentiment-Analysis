import os
from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.linalg import Vectors
from sklearn.feature_extraction.text import HashingVectorizer
from scipy.sparse import csc_matrix

conf = SparkConf().setAppName('sentimentAnalysis').setMaster('local')
sc = SparkContext(conf=conf)



amazon_dataset = sc.textFile('all.txt')

subset_size = 34686770

reviews = amazon_dataset.filter(lambda l: 'review/text:' in l).map(lambda l: l[13:])#.take(subset_size))#.sample(False, 0.01, seed=42)
sentiments = amazon_dataset.filter(lambda l: 'review/score:' in l).map(lambda l: 1 if float(l[14:]) > 3.0 else 0)#.take(subset_size))#.map(lambda l: float(l[14:])*2).take(subset_size))#.sample(False, 0.01, seed=42)

indexed_reviews = reviews.zip(sc.parallelize([n for n in range(subset_size)]))
indexed_sentiments = sentiments.zip(sc.parallelize([n for n in range(subset_size)]))

train_reviews = indexed_reviews.filter(lambda r: r[1]%2 == 0)
train_sentiments = indexed_sentiments.filter(lambda r: r[1]%2 == 0)

hv = HashingVectorizer(binary=True, ngram_range=(1,3))
hv_global = sc.broadcast(hv)
train_reviews.map(lambda r: hv_global.value.partial_fit(r[0]))

test_reviews = indexed_reviews.filter(lambda r: r[1]%2 == 1)
test_sentiments = indexed_sentiments.filter(lambda r: r[1]%2 == 1)


def vectorizer(text):
	'''
	takes in a dictionary of the vocabulary and makes a sparse feature vector where the features are word occurence
	'''
	indices = hv_global.value.transform([text]).nonzero()
	return Vectors.sparse(2**20, indices[1], [1 for n in range(len(indices[0]))])

train_sentiments_noindex = train_sentiments.map(lambda s: s[0])
review_vectors = train_reviews.map(lambda r: vectorizer(r[0]))
training_vectors = train_sentiments_noindex.zip(review_vectors).map(lambda g: LabeledPoint(g[0], g[1]))

clf = NaiveBayes.train(training_vectors)

predictions = test_reviews.map(lambda r: clf.predict(vectorizer(r[0])))

num_correct = float(predictions.zip(test_sentiments).filter(lambda x: x[0] == x[1][0]).count())
num_total = subset_size/2

print num_correct, num_total, num_correct/num_total