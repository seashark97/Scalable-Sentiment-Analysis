from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes

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

reviews = amazon_dataset.filter(lambda l: 'review/text:' in l)
sentiments = amazon_dataset.filter(lambda l: 'review/score:' in l).map(lambda l: 1.0 if float(l[-3:]) > 2.5 else 0.0)

train_text = sc.parallelize(reviews.take(20)[:10])#[:int(reviews.count()*0.75)]) taking first 10 elements for local testing    #taking the first 75% of the reviews
test_text = sc.parallelize(reviews.take(20)[10:])#[int(reviews.count()*0.75):])

train_sentiments = sc.parallelize(sentiments.take(20)[:10])#[:int(sentiments.count()*0.75)])
test_sentiments = sc.parallelize(sentiments.take(20)[10:])#[int(sentiments.count()*0.75):])

vocab = train_text.flatMap(ngram_range).distinct()
vocab_dict = dict(vocab.zip(sc.parallelize([0 for n in range(vocab.count())])).collect())    #vocabulary dictionary. Format:  {'n-gram':0 for every n-gram in training set}
# vocab_dict.cache()
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
# train_text_vectors = train_text.map(vectorizer)
training_vectors = sc.parallelize(zip(train_sentiments.collect(), train_text.map(vectorizer).collect())).map(LabeledPoint).collect()
test_vectors = test_text.map(vectorizer)

clf = NaiveBayes.train(sc.parallelize(training_vectors))
predictions = clf.predict(test_vectors)

accuracy = predictions.zip(test_sentiments).filter(lambda x: x[0] != x[1]).count()/predictions.count()

print accuracy