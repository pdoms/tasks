import csv
import os
import re    
import numpy as np
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation
from nltk.stem import PorterStemmer, WordNetLemmatizer
from num2words import num2words

def is_number_regex(s):
    if re.match("^\d+?\.\d+", s) is None:
        return s.isdigit()
    return True


class Tokens():
    def __init__(self):
        self._data = []
        self.vocab = {}
        self._index = 0
        self.sparse = []
        self._tokens = []
        self.count_vector = []
        self.token_list = []
        self.stopwords =[]
        self.preprocess_opts = ['lower', 'num2word', 'stopwords', 'punctuation', 'lemmatization', 'stem', 'symbols']

    def __call__(self, data, encoding='utf-8', column=None, preprocess=[], symbols=[]):
        if isinstance(data, list):
            self._data = data
        elif data.endswith('csv'):
            self._from_cvs(data, column)
        elif data.endswith('txt'):
            self._from_txt(data, encoding)
        else: 
            raise ValueError('Pass either a List or a path to a csv or txt file.')
        self._preprocess(preprocess, symbols)
        self._create_vocab()
        self._vectorize_sparse()

    def get_tokens(self):
        return np.array(self._tokens)

    def _from_cvs(self, data, column=None):
        if column is None:
            raise ValueError('Pass a column index as int')
        with open(data) as file:
            reader = csv.reader(file)
            self._data = np.array([row[column] for row in reader])

    def _from_txt(self, data, encoding):
        with open(data, 'r', encoding=encoding) as file:
            self._data = np.array([row for row in file.readlines()])
            
    def _create_vocab(self):
        for sentence in self._data:
            tkns = sentence.split(' ')
            for t in tkns:
                if t not in self._tokens:
                    self._tokens.append(t)
            listed = []
            for word in tkns:
                if word not in self.vocab and word != '':
                    listed.append(word)
                    self.vocab[word] = self._index
                    self._index += 1
            self.token_list.append(listed)

    def expand_vocab(self, data, encoding='utf-8', column=None):
        pass

    def _preprocess(self, options=[], symbols=[]):
        if len(options) == 0:
            options = self.preprocess_opts
        for opt in options:
            for i, sentence in enumerate(self._data):
                if opt == 'lower':
                    sentence = self._to_lower(sentence)
                if opt == 'num2word':
                    sentence = self._num_to_words(sentence)
                if opt == 'stopwords':
                    sentence = self._remove_stopwords(sentence)
                if opt == 'punctuation':
                    sentence = self._remove_punctuation(sentence)
                if opt == 'lemmatization':
                    sentence = self._lemmatize(sentence)
                if opt == 'stem':
                    sentence = self._stem(sentence)
                if opt == 'symbol':
                    if len(symbols)== 0:
                        print('Preprocessing: removing custom symbols was requested, but no symbols were provided... skipping the step.')
                    else:
                        sentence = self._custom_symbols(sentence)
            self._data[i] = sentence
        return self._data


    def _to_lower(self, sentence):
        return sentence.lower()
    
    def _remove_stopwords(self, sentence):
        return remove_stopwords(sentence)
    
    def _remove_punctuation(self, sentence):
        return strip_punctuation(sentence)
    

    def _lemmatize(self, sentence):
        clean = []
        lemmatizer = WordNetLemmatizer()
        for word in sentence.split(' '):
            word = lemmatizer.lemmatize(word)
            clean.append(word)
        return ' '.join(clean)

    def _stem(self, sentence):
        clean = []
        stemmer = PorterStemmer()
        for word in sentence.split(' '):
            word = stemmer.stem(word)
            clean.append(word)
        return ' '.join(clean)

    def _num_to_words(self, sentence):
        clean = []
        for word in sentence.split(' '):
            if is_number_regex(word):
                word = num2words(word)
            clean.append(word)
        return ' '.join(clean)


    def _custom_symbols(self, sentence, symbols):
        for s in symbols:
            sentence = sentence.translate(str.maketrans('', '', s))
        return sentence



    def _vectorize_sparse(self):
        n = len(self._data)
        self.count_vector = np.zeros((n, len(self.vocab)))
        for i, sentence in enumerate(self.token_list):
            for word in sentence:
                    self.count_vector[i][self.vocab[word]] += 1
    
    

p = os.getcwd()
p = p.replace('src', '')
p = os.path.join(p, 'data', 'Mini_Tweets', 'negative.csv')
t = Tokens()
t(p, column=5)





















