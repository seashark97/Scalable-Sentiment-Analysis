from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.linalg import SparseVector

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

reviews = amazon_dataset.filter(lambda l: 'review/text:' in l).map(lambda l: l[13:])
sentiments = amazon_dataset.filter(lambda l: 'review/score:' in l).map(lambda l: 1.0 if float(l[14:]) > 2.5 else 0.0)

train_text = sc.parallelize(reviews.take(2000)[:1000])#[:int(reviews.count()*0.75)]) taking first 10 elements for local testing    #taking the first 75% of the reviews
test_text = sc.parallelize(reviews.take(2000)[1000:])#[int(reviews.count()*0.75):])

train_sentiments = sc.parallelize(sentiments.take(2000)[:1000])#[:int(sentiments.count()*0.75)])
test_sentiments = sc.parallelize(sentiments.take(2000)[1000:])#[int(sentiments.count()*0.75):])

vocab = train_text.flatMap(ngram_range).distinct().cache().collect()
# vocab_dict = dict(vocab.zip(sc.parallelize([0 for n in range(vocab.count())])).collect())    #vocabulary dictionary. Format:  {'n-gram':0 for every n-gram in training set}
# vocab_dict.cache()
def vectorizer(text, vocab=vocab):
	'''
	takes in a dictionary of the vocabulary and makes a binary feature vector where the features are word occurence
	'''
	i=0
	nonzero = []
	words = ngram_range(text)
	
	for word in vocab:
		if word in words:
			nonzero.append((i, 1))
		# else:
		# 	nonzero.append((i, 0))
		i+=1
	return SparseVector(i, nonzero)
	# words = ngram_range(text)
	# copy = vocab_dict.copy()
	# for word in words:
	# 	if word in copy:
	# 		copy[word] = 1
	# vector = []
	# return copy.values()

# train_text_vectors = train_text.map(vectorizer)
print '#########################################################'
print 'Checkpoint 1'
training_vectors = train_sentiments.zip(train_text.map(vectorizer)).map(lambda labeled: LabeledPoint(labeled[0], labeled[1])).collect()
print 'Checkpoint 1.5'
clf = NaiveBayes.train(training_vectors)
print '#########################################################'
print 'Checkpoint 2'
test_vectors = test_text.map(vectorizer)
predictions = test_vectors.map(clf.predict)
print predictions.collect()
print test_sentiments.collect()
num_correct = float(predictions.zip(test_sentiments).filter(lambda x: x[0] == x[1]).count())
num_total = float(predictions.count())

print num_correct, num_total, num_correct/num_total