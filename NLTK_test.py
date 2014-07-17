import nltk
import nltk.data
from nltk import bigrams, trigrams
text = open('/Users/abdul/Desktop/RSI/test_code/BB.txt').read()

'''
class readyText(object):
	def __init__(self, text):
		self.text = text
		self.splitText = split(self.text)
		self.posTagged = posTag(self.splitText)
	def split(self):
		return self.splitText
	def posTag(self):
		return self.posTagged
	def __str__(self):
		return self.text

def split(text):
	splitter = nltk.data.load('tokenizers/punkt/english.pickle')
	tokenizer = nltk.tokenize.TreebankWordTokenizer()
	sentences = splitter.tokenize(text)
	tokenized_sentences = [tokenizer.tokenize(sent) for sent in sentences]
	return tokenized_sentences

def posTag(sentences):
	pos = [nltk.pos_tag(sentence) for sentence in sentences]
	return pos

BBspeech = readyText(text)

#makeVerbGraph(posTag(split(text)))
#print BBspeech
#print BBspeech.posTag()


unigrams = nltk.word_tokenize(text)
bigrams = bigrams(unigrams)
trigrams = trigrams(unigrams)

print unigrams
print bigrams
print trigrams
'''
