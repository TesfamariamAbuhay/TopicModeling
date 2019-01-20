# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 15:51:42 2016

@author: user
"""

import numpy as np
import matplotlib.pylab as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os
from sklearn import decomposition
import unsupervised.nfm
from optparse import OptionParser

corpus_path = "D:\\Topic Modeling\\preprocessed corpus\\2010\Corpus\\TMfor15years\\"
filenames = sorted([os.path.join(corpus_path,fn) for fn in os.listdir(corpus_path)])
        
vectorizer = CountVectorizer(input='filename', stop_words='english', min_df = 16)
dtm = vectorizer.fit_transform(filenames).toarray()
vocab = np.array(vectorizer.get_feature_names())

tfidf_vectorizer = TfidfVectorizer(input = 'filename', stop_words = 'english')
tfidf = tfidf_vectorizer.fit_transform(filenames).toarray()

num_topics = 16
num_top_words = 20
clf = decomposition.NMF(n_components=num_topics, random_state=1)
doctopic = clf.fit_transform(dtm)

nmf = decomposition.NMF(n_components=num_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)

def print_top_words(model, feature_names, num_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-num_top_words - 1:-1]]))
    print()
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, num_top_words)

topic_words = []

for topic in clf.components_:
    word_idx = np.argsort(topic)[::-1][0:num_top_words]
    topic_words.append([vocab[i] for i in word_idx])

N, K = doctopic.shape
ind = np.arange(N)
year = [2001, 2002, 2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015, 2016]
width = 1
height = 1
plots = []
height_cumulative = np.zeros(N)
for k in range(K):
    color = plt.cm.coolwarm(k/K, 1)
    if K == 0:
        p = plt.bar(year, doctopic[:,k], width, height, color=color)
    else:
        p = plt.bar(year, doctopic[:,k], width, bottom = height_cumulative, color = color)
    height_cumulative += doctopic[:,k]
    plots.append(p)

#plt.ylim((0,1))
plt.ylabel("Proportion of Topics' Per Year")
plt.xlabel("Year")
plt.xlim(2001,2016)
plt.title("Topics in ICCS")
#plt.xticks(ind+width/2)
#plt.yticks(np.arange(0,1,10))
topic_labels = ['Topic #{}'.format(k) for k in range(K)]
plt.legend([p[0] for p in plots], topic_labels, loc = 0)
plt.show()
        