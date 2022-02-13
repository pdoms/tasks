import csv
import os
import re    
import numpy as np
import copy
import math
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, STOPWORDS
from nltk.stem import PorterStemmer, WordNetLemmatizer
from num2words import num2words
from distances import cosine_distance
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

#TODO 
#check more of stemming and lemmatization (not happy)
#finish query
#csr 
#support vector classifier
#word embeddings

def is_number_regex(s):
    if re.match("^\d+?\.\d+", s) is None:
        return s.isdigit()
    return True


class Words():
    def __init__(self):
        self.data = []
        self.n = 0
        self.vocab = {}
        self.index = 0
        self.tokenized = []
        self.count_vector = []
        self.df = []
        self.idf = []
        self.tf_idf = []
        self.preprocess_opts = ['whitespace', 'lower', 'num2word', 'lemmatization', 'stem', 'punctuation', 'stopwords', 'single_character', 'symbols']
        self.last_preproc_opts = ()
        self.is_query = False


    def __call__(self, data, encoding='utf-8', column=None, preprocess=[], symbols=[]):
        if isinstance(data, list):
            self.data = data
        elif data.endswith('csv'):
            self.from_cvs(data, column)
        elif data.endswith('txt'):
            self.from_txt(data, encoding)
        else: 
            raise ValueError('Pass either a List or a path to a csv or txt file.')
        self.preprocess(preprocess, symbols)
        self.n = len(self.data)
        self.tokenize()
        self.create_vocab()
        self.count_vectorize()
        self.document_freq()
        self.inv_doc_freq()
        self.calc_tf_idf()
        return self.tf_idf


    
    def tokenize(self):
        for sentence in self.data:
            self.tokenized.append(sentence.split(' '))
    
    def create_vocab(self):
        for sentence in self.tokenized:
            for word in sentence:
                if word not in self.vocab:
                    self.vocab[word] = self.index
                    self.index += 1
    
    def count_vectorize(self):
        self.count_vector = np.zeros((self.n, len(self.vocab)))
        for i, document in enumerate(self.tokenized):
            for token in document:
                self.count_vector[i][self.vocab[token]] += 1
    
    def document_freq(self):
        self.df = np.count_nonzero(self.count_vector > 0, axis=0)
    
    def inv_doc_freq(self):
        self.idf = np.log(self.n/self.df) + 1
    
    def calc_tf_idf(self, normalized=True):
        self.tf_idf = np.copy(self.count_vector*self.idf)
        if normalized:
            self.unit_norm_l1()
    
    def unit_norm_l1(self):
        for i, doc in enumerate(self.tf_idf):
            tot = np.sum(doc)
            for j, token in enumerate(doc):
                self.tf_idf[i][j] = token/tot

    def query(self, query):
        pass

    def cosine_matrix(self):
        pass
        
     
        


    
       















    def from_cvs(self, data, column=None):
        if column is None:
            raise ValueError('Pass a column index as int')
        with open(data) as file:
            reader = csv.reader(file)
            self.data = np.array([row[column] for row in reader])
  

    def from_txt(self, data, encoding):
        with open(data, 'r', encoding=encoding) as file:
            self.data = np.array([row for row in file.readlines()])
            


    def preprocess(self, options=[], symbols=[], query=None):
        query_data = query
        use_data = []
        if len(options) == 0:
            options = self.preprocess_opts
        if query is not None:
            if len(self.last_preproc_opts[0]) == 0:
                raise ValueError('Cannot use stored options, please provide preprocessing options to match the vocab') 
            else: 
                options, symbols = self.last_preproc_opts
                use_data = query
        else:        
            self.last_preproc_opts = (options, symbols)
            use_data = self.data

        for opt in options:
            for i, sentence in enumerate(use_data):
                if opt == 'whitespace':
                    sentence = self.remove_whitespaces(sentence)
                if opt == 'lower':
                    sentence = self.to_lower(sentence)
                if opt == 'num2word':
                    sentence = self.num_to_words(sentence)
                if opt == 'stopwords':
                    sentence = self.remove_stopwords(sentence)          
                if opt == 'punctuation':
                    sentence = self.remove_punctuation(sentence)
                if opt == 'lemmatization':
                    sentence = self.lemmatize(sentence)
                if opt == 'stem':
                    sentence = self.stem(sentence)
                if opt == 'single_character':
                    sentence = self.sngl_char(sentence)
                if opt == 'symbol':
                    if len(symbols)== 0:
                        print('Preprocessing: removing custom symbols was requested, but no symbols were provided... skipping the step.')
                    else:
                        sentence = self.custom_symbols(sentence)
                if query is not None:
                    query_data[i] = sentence
                else: 
                    self.data[i] = sentence
        if query is None:
            return self.data
        else: 
            return query_data

    def remove_whitespaces(self, sentence):
        clean = []
        for word in sentence.split(' '):
            clean.append(word.strip())
        return ' '.join(clean)

    def to_lower(self, sentence):
        return sentence.lower()
    
    def remove_stopwords(self, sentence):
        return remove_stopwords(sentence)
    
    def remove_punctuation(self, sentence):
        return strip_punctuation(sentence)
    

    def lemmatize(self, sentence):
        clean = []
        lemmatizer = WordNetLemmatizer()
        for word in sentence.split(' '):
            word = lemmatizer.lemmatize(word)
            clean.append(word)
        return ' '.join(clean)

    def stem(self, sentence):
        clean = []
        stemmer = PorterStemmer()
        for word in sentence.split(' '):
            word = stemmer.stem(word)
            clean.append(word)
        return ' '.join(clean)

    def num_to_words(self, sentence):
        clean = []
        for word in sentence.split(' '):
            if is_number_regex(word):
                word = num2words(word)
            clean.append(word)
        return ' '.join(clean)

    def sngl_char(self, sentence):
        no_sngl = []
        for word in sentence.split(' '):
            if len(word) > 1:
                no_sngl.append(word)
        return ' '.join(no_sngl)


    def custom_symbols(self, sentence, symbols):
        for s in symbols:
            sentence = sentence.translate(str.maketrans('', '', s))
        return sentence



  
                
                

                

        
        
                

        
        
        


    #below are compressed sparse row matrix variations - prefixed with csr
            

#p = os.getcwd()
#p = p.replace('src', '')
#p = os.path.join(p, 'data', 'Mini_Tweets', 'negative.csv')

corpus = [
'Data is the oil of the digital economy',
'Data is a new oil'
]

w = Words()
tf_idf = w(corpus, preprocess=['lower', 'punctuation', 'single_character'])
print(list(w.vocab.keys()))
print('\n\n')
print(tf_idf)
print('\n\n')

v = TfidfVectorizer(norm='l1')
fit = v.fit_transform(corpus)
print(v.get_feature_names_out())
print('\n\n')
print(fit.toarray())




































