# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 12:19:09 2016

@author: user
"""

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#from nltk.corpus import stopwords
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
#from nameparser import HumanName
#from nltk import stem
import gensim
from gensim import corpora, models
import os
path='D:\\Topic Modeling\\preprocessed corpus\\2010\\Corpus\\2015'
all_docs = []
for article in os.listdir(path):
    articlepath=os.path.join(path,article)
    files=open(articlepath,'r')
    doc = files.read()
    all_docs.append(doc)

tokenizer = RegexpTokenizer(r'\b([a-zA-Z]+)\b')
token = []
for doc in all_docs:
    doc = doc.lower()
    token.append(tokenizer.tokenize(doc))    

 
stopw = open('D:\\Topic Modeling\\stop words.txt', 'r')
stop_words = stopw.read()
  
texts = [[word for word in document if word not in stop_words]
         for document in token]

remove = open('D:\\Topic Modeling\\remove.txt', 'r')
remove = remove.read()
texts = [[word for word in document if word not in remove]
         for document in texts]
           
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1]
        for text in texts]
texts = [[token for token in text if len(token) > 2]
        for text in texts]

texts = [[token for token in text if len(token) < 30]
        for text in texts]
            
lemmatizer = WordNetLemmatizer()
lemma_text = []     
for word in texts:
    lemma = [lemmatizer.lemmatize(i) for i in word]
    lemma_text.append(lemma)  

stemmer = PorterStemmer()
stem_text = []
for word in texts:
    stemm = [stemmer.stem(i) for i in word]
    stem_text.append(stemm)   

#without lemmatizer and stemmer
dictionary_text = corpora.Dictionary(texts)
dictionary_text.save('D:\\Topic Modeling\\preprocessed corpus\\2010\\Corpus\\result_text2015.dict')
result_text = open('D:\\Topic Modeling\\preprocessed corpus\\2010\\Corpus\\result_text2015.txt', 'w')
result_text.write(str(dictionary_text.token2id))
result_text.close()
corpus_text = [dictionary_text.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('D:\\Topic Modeling\\preprocessed corpus\\2010\\Corpus\\result_text2015.mm', corpus_text)
mm = gensim.corpora.MmCorpus('D:\\Topic Modeling\\preprocessed corpus\\2010\\Corpus\\result_text2015.mm')
lda_text = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=dictionary_text, num_topics=44, passes=200)
lda_text.top_topics(corpus_text, num_words=10)

lda_corpus = lda_text[corpus_text]
for doc in lda_corpus:
    print (doc)

print (dictionary_text)

lda_text.get_document_topics(corpus_text)
lda_text.show_topics(num_topics = 10, num_words = 10, log = True, formatted = True)

print(dictionary_text)

# using lemmatized corpus as input    
dictionary_lema = corpora.Dictionary(lemma_text)
dictionary_lema.save('D:\\Topic Modeling\\preprocessed corpus\\2010\\Corpus\\result_lema2010.dict')
result_lema = open('D:\\Topic Modeling\\preprocessed corpus\\2010\\Corpus\\result_lema2010.txt', 'w')
result_lema.write(str(dictionary_lema.token2id))
result_lema.close()
corpus_lema = [dictionary_lema.doc2bow(text) for text in lemma_text]
corpora.MmCorpus.serialize('D:\\Topic Modeling\\preprocessed corpus\\2010\\Corpus\\result_lema2010.mm', corpus_lema)
mm_lema = gensim.corpora.MmCorpus('D:\\Topic Modeling\\preprocessed corpus\\2010\\Corpus\\result_lema2010.mm')
lda_lema = gensim.models.ldamodel.LdaModel(corpus=mm_lema, id2word=dictionary_lema, num_topics=18, update_every=0, passes=200)
lda_lema.top_topics(corpus_lema, num_words = 10)

result_text = open('D:\\Topic Modeling\\preprocessed corpus\\2010\\Corpus\\top_topics_2007.txt', 'w')
result_text.write(str(lda_lema.top_topics(corpus_lema, num_words = 10)))
result_text.close()

#using stemmemized corpus as input
dictionary_stem = corpora.Dictionary(stem_text)
dictionary_stem.save('D:\\Topic Modeling\\preprocessed corpus\\2010\\Corpus\\result_stem2007.dict')
result_stem = open('D:\\Topic Modeling\\preprocessed corpus\\2010\\Corpus\\result_stem2007.txt', 'w')
result_stem.write(str(dictionary_stem.token2id))
result_stem.close()
corpus_stem = [dictionary_stem.doc2bow(text) for text in stem_text]
corpora.MmCorpus.serialize('D:\\Topic Modeling\\preprocessed corpus\\2010\\Corpus\\result_stem2007.mm', corpus_stem)
mm_stem = gensim.corpora.MmCorpus('D:\\Topic Modeling\\preprocessed corpus\\2010\\Corpus\\result_stem2007.mm')
lda_stem = gensim.models.ldamodel.LdaModel(corpus=mm_stem, id2word=dictionary_stem, num_topics=18, passes=200)
lda_stem.top_topics(corpus_lema, num_words = 10)


lda_stem2 = gensim.models.ldamodel.LdaModel(corpus=corpus_stem, id2word=dictionary_stem, num_topics=10, passes=20)
lda_stem2.print_topics(10)
