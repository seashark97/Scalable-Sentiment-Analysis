from pyspark import SparkContext, SparkConf
import string
from sklearn.feature_extraction.text import HashingVectorizer
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes

conf = SparkConf().setAppName(sentimentAnalysis).setMaster(local)
sc = SparkContext(conf=conf)

def ngram_range(text):
	'''
	returns 1, 2, 3-grams without punctuation
	'''
	exclude = set(string.punctuation)
	words = ''.join(ch for ch in text if ch not in exclude).lower().split()
	bigrams = [' '.join([words[n-1], words[n]]) for n in range(1,len(words))]
	trigrams = [' '.join([words[n-2], words[n-1], words[n]]) for n in range(2,len(words))]
	return words + bigrams + trigrams



amazon_dataset = sc.textFile('all.txt')

reviews = amazon_dataset.filter(lambda l: 'review/text:' in l)
sentiments = amazon_dataset.filter(lambda l: 'review/score:' in l).map(lambda l: 1.0 if float(l[-3:]) > 2.5 else 0.0)

train_text = sc.parallelize(reviews.collect()[:10])#[:int(reviews.count()*0.75)]) taking first 10 elements for local testing    #taking the first 75% of the reviews
test_text = sc.parallelize(reviews.collect()[10:20])#[int(reviews.count()*0.75):])

train_sentiments = sc.parallelize(sentiments.collect()[:10])#[:int(sentiments.count()*0.75)])
test_sentiments = sc.parallelize(sentiments.collect()[10:20])#[int(sentiments.count()*0.75):])

vocab = train_text.flatmap(ngram_range).distinct()
vocab_dict = dict(vocab.zip([0 for n in range(vocab.count())]))    #vocabulary dictionary. Format:  {'n-gram':0 for every n-gram in training set}
vocab_dict.persist()
def vectorizer(text, vocab=vocab_dict):
	'''
	takes in a dictionary of the vocabulary and makes a binary feature vector where the features are word occurence
	'''
	words = ngram_range(text)
	vocab_copy = vocab.copy()
	for word in words:
		if word in vocab_copy:
			vocab_copy[word] = 1
	return vocab_copy.values()

training_vectors = train_sentiments.zip(train_text.map(vectorizer)).map(LabeledPoint)
test_vectors = test_text.map(vectorizer)

clf = NaiveBayes.train(training_vectors, 1.0)
predictions = clf.predict(test_vectors)

accuracy = predictions.zip(test_sentiments).filter(lambda x: x[0] != x[1]).count()/predictions.count()

print accuracy