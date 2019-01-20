# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 09:25:12 2016

@author: user
"""
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
import re
import codecs
import seaborn as sns
import unsupervised.nmf, unsupervised.rankings
#Rename multiple files
path = ('Documents/Data/ICCS_2015')
files = os.listdir(path)
order = 000
for file in files:
    os.rename(os.path.join(path,file), os.path.join(path,'ICCS_2015_' + str(order)))
    order = order + 1
#
dir_data = "D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Data"
filenames = sorted([os.path.join(dir_data,fn) for fn in os.listdir(dir_data)])
stopwords = open("D:\\Topic Modeling\\stop words.txt", 'r')
stopwords = stopwords.read()
stopwords = stopwords.split()
doc_id = sorted(os.listdir(dir_data))

#file_paths = [os.path.join(dir_data, fname) for fname in os.listdir(dir_data) if fname.endswith(".txt") ]
#documents = [codecs.open(file_path, 'r', encoding="utf8").read() for file_path in file_paths ]
token_pattern = re.compile(r"\b\w\w+\b", re.U)
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
topic_terms = []
doc_rankings1 = []

def custom_tokenizer( s, min_term_length = 2 , max_term_length = 30):
    res = [x.lower() for x in token_pattern.findall(s) if (len(x) > min_term_length and len(x) <=max_term_length and x[0].isalpha() ) ]
    res = [token for token in res if token not in stopwords]
    return [stemmer.stem(plural) for plural in res]

def preprocess( filenames, tokenizer=custom_tokenizer ):
    tfidf = TfidfVectorizer(input = 'filename', stop_words='english',lowercase=True, strip_accents="unicode", tokenizer=tokenizer, use_idf=True, norm="l2", min_df = 10, ngram_range = (1,1)) 
    A = tfidf.fit_transform(filenames)
    terms = []
	
    num_terms = len(tfidf.vocabulary_)
    terms = [""] * num_terms
    for term in tfidf.vocabulary_.keys():
        terms[ tfidf.vocabulary_[term] ] = term
    #
    model = decomposition.NMF(init="nndsvd", n_components=150, max_iter=800)
    W = model.fit_transform(A)
    H = model.components_	
    #
    
    for topic_index in range( H.shape[0] ):
        top_indices = np.argsort( H[topic_index,:] )[::-1][0:10]
        term_ranking = [terms[i] for i in top_indices]
        topic_terms.append(term_ranking)
        print ("Topic %d: %s" % ( topic_index, ", ".join( term_ranking ) ))
    
    volume_names = ['01', '02', '03', '04', '05', '06', '07', '08', '09','10', '11', '12', '13', '14', '15', '16']
    volume_indexes = []
    for volname in volume_names:
        for i, paper_name in enumerate(filenames):
            if volname in paper_name:
                volume_indexes.append(i)
                break
    series_smooth2 = []
    for i in range(150):
        plt.clf()  # clears the current plot
        plt.figure(figsize = (18.5, 10.5))
        series = W[:, i]
        xs = np.arange(len(series))
        text_xs = [0, 252, 433, 965, 1527, 1984, 2601, 3308, 3584, 3786, 4095, 4350, 4574, 4870, 5107, 5440, 5965]#np.array(volume_indexes) + np.diff(np.array(volume_indexes + [max(xs)]))/2
        text_ys = np.repeat(max(series), len(volume_names)) - 0.05
        #n = [253, 184, 532, 562, 457, 617, 707, 276, 202, 309, 255, 224, 296, 237, 333, 255]
        series_smooth = pd.rolling_mean(series, 100)
        series_smooth2.append([])
        f = 0
        while f in range(16):
            series_smooth2[i].append(np.mean(series[text_xs[f]:text_xs[f+1]]))
            f = f + 1
        p1 = plt.plot(series, '.')
        p2 = plt.plot(series_smooth2[i], '-', linewidth=2)
        plt.vlines(text_xs, ymin=0, ymax=np.max(series))
        for x, y, s in zip(text_xs, text_ys, volume_names):
            plt.text(x, y, s)
        plt.title("Topic {}: {}".format(i, topic_terms[i]))
        plt.ylabel("Topic share(0-1)")
        plt.xlabel("ICCS papers segemented per year")
        plt.ylim(0, max(series))
        plt.xlim(0, 5965)
        plt.legend((p1[0],p2[0]), ('topic share', 'moving average'))
        savefig_fn = "D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Plots\\ICCS-stem-K150-topic{}.png".format(i)
        plt.savefig(savefig_fn, format='png')
    MA = pd.DataFrame(series_smooth2)
    MA.to_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Plots\\MAFinal1502.csv")
    TTT = pd.DataFrame(topic_terms)
    TTT.to_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Plots\\TTTFinal1502.csv")
    
    k = W.shape[1]
    for topic_index in range(k):
        w = np.array( W[:,topic_index] )
        top_indices = np.argsort(w)[::-1]
        doc_rankings1.append(top_indices)
        
    #base_name = os.path.splitext( os.path.split(filenames)[-1] )[0]
    doc_rankings = unsupervised.nmf.generate_doc_rankings( W )
    actual_top = min(10,len(filenames))
    #doc_path = "D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Plots"
    #log.info("Writing top %d document IDs to %s" % (options.top,doc_path) )
    fout = codecs.open("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Plots\\doc_topic1502.csv", "w", "utf-8")
    fout.write("Rank")
    for i in range(150):
        fout.write(" ,Topic%s" % i)
    fout.write("\n")
    for pos in range(actual_top):
        fout.write( "%d" % (pos + 1) )
        for ranking in doc_rankings:
            fout.write( ",%s" % doc_id[ranking[pos]])
        fout.write("\n")
    fout.close()
    return;
preprocess(filenames)




dir_out = ("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Plots")
doc_rankings = []
k = W.shape[1]
for topic_index in range(k):
    w = np.array( W[:,topic_index] )
    top_indices = np.argsort(w)[::-1]
    doc_rankings.append(top_indices)

doc_rankings = unsupervised.nmf.generate_doc_rankings( W )
actual_top = min(10,len(filenames))
doc_path = os.path.join(dir_out, "%s_top%d_docids.csv"  % (base_name,10))
log.info("Writing top %d document IDs to %s" % (options.top,doc_path) )
fout = codecs.open(doc_path, "w", "utf-8")
fout.write("Rank")
for label in labels:
    fout.write(",%s" % label )
fout.write("\n")
for pos in range(actual_top):
    fout.write( "%d" % (pos + 1) )
    for ranking in doc_rankings:
        fout.write( ",%s" % doc_ids[ranking[pos]])
    fout.write("\n")
fout.close()


Data = pd.read_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Plots\\Average Topic Share per Year.csv")
data = Data.iloc[:,[0,3]]
columns = columns[:2]
plt.figure(figsize = (15, 10))
color = ['b', 'g', 'r', 'c','m','y', 'k', 'indigo', 'gold', 'hotpink','firebrick', 'indianred', 'sage', 'coral', 'powderblue']
r = 0
for i in columns[30:40]:
    p= sns.plt.plot(Data.Year, Data[i], 'o-', color = color[r])
    sns.plt.legend(topic_terms[30:40], fontsize=14, loc = 'center left', bbox_to_anchor = (1,0.5))
    r =r + 1
sns.plt.xlim([1,17])
sns.plt.title("Dynamics of Topics")
sns.plt.xlabel("Year")
sns.plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose
data2 = Data['Year'], Data['Topic 1']
decomposition = seasonal_decompose(Data['Topic 1'], model='additive')

for i in columns[:10]:
    g = sns.PairGrid(Data)
    g.map(plt.scatter)

data = pd.read_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Plots\\coretwo.csv")
doc_id = pd.read_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Plots\\doc_topic.csv")

g = sns.PairGrid(data)
g.map(sns.regplot)
g.map(scipy.stats.pearsonr(data[1],data[2]))

data_columns = list(data.columns) 
core150 = []
for i in data_columns:
    t = data_columns.index(i) + 1
    while t in range(96):
        r = scipy.stats.pearsonr(data[i],data[data_columns[t]])
        core150.append([i,data_columns[t], r])
        t =t + 1

for i in data_columns:
    t = data_columns.index(i) + 1
    while t in range(96):
        for d in doc_id[i]:
            if d in doc_id[data_columns[t]]:
                 #r = scipy.stats.pearsonr(data[i],data[data_columns[t]])
                 core150.append([i,data_columns[t]])
        t =t + 1
        
        
sns.regplot(data['Topic 22'], data['Topic 67'])
sns.plt.title(scipy.stats.pearsonr(data['Topic 22'], data['Topic 67']))
Data2 = pd.DataFrame(core150)
Data2.to_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Plots\\core150.csv")
Data.plot().legend(loc = 'center left', bbox_to_anchor = (1,0.5))
t = pd.DataFrame(topic_terms)
t.savetxt("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Plots\\TT.txt")
TTT = pd.DataFrame(topic_terms)
TTT.to_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Plots\\TTTFinal.csv")
TTT = [x.decode('utf-8') if isinstance(x, str) else x for x in topic_terms]
ta = t.drop(t.index[38])
ta = ta.T

doc_id = pd.read_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Plots\\doc_topic.csv")
doc_id_col = list(doc_id.columns) 
for i in doc_id:
    for l in i:
        if l in doc_id[i]




filet = codecs.open("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Plots\\TTT.csv", "w", "utf-8")
for i in topic_terms:
    for l in i:
        filet.write(l)

#plt.legend((p1[0],p2[0]),p3[0] ('Topic 0', 'Topic 1', 'Topic 2'))
plt.show()
Data2.to_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Plots\\MA2.csv")
N, K = W.shape
ind = np.arange(N)
ind = ind[:15]
#year = [2001, 2002, 2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015, 2016]
width = 1
plots = []
height_cumulative = np.zeros(N)
height_cumulative = height_cumulative[:15]
r=0
for k in range(K):
      color = ['b', 'g', 'r', 'c','m','y', 'k','w', 'indigo', 'gold', 'hotpink','firebrick', 'indianred', 'sage', 'coral', 'powderblue']
      p = plt.bar(ind, W[0:15, k], width, bottom=height_cumulative, color=color[r])
      height_cumulative += W[0:15, k]
      plots.append(p)
      r= r +1

plt.ylim((0,1))
plt.ylabel("Proportion of Topics' Per Paper")
plt.xlabel("Papers of ICCS 2016")
plt.xlim(0,15)
plt.title("Topics in ICCS 2016")
plt.xticks(ind+width/2)
plt.yticks(np.arange(0,1,10))
topic_labels = ['Topic #{}'.format(k) for k in range(K)]
plt.legend(topic_labels, loc = 0)
plt.show()

######
N, K = W.shape
ind = np.arange(N)
ind = ind[:15]
#year = [2001, 2002, 2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015, 2016]
width = 1
plots = []
height_cumulative = np.zeros(N)
height_cumulative = height_cumulative[:]
for k in range(N):
    color = plt.cm.coolwarm(k/N,1)
    #color = ['b', 'g', 'r', 'c','m','y', 'k','w', 'indigo', 'gold', 'hotpink','firebrick', 'indianred', 'sage', 'coral', 'powderblue']
    p = plt.bar( W[:, k], ind,width, bottom=height_cumulative, color=color)
    height_cumulative += W[:, k]
    plots.append(p)
    r= r +1

plt.Xlim((0,1))
plt.ylabel("Proportion of Topics' Per Paper")
plt.xlabel("Papers of ICCS 2016")
plt.xlim(0,15)
plt.title("Topics in ICCS 2016")
plt.xticks(ind+width/2)
plt.yticks(np.arange(0,1,10))
topic_labels = ['Doc #{}'.format(k) for k in range(N)]
plt.legend(topic_labels, loc = 0)
plt.show()


plt.bar(ind, W[:15,0], color= 'red')
plt.bar(ind, W[:15,1], color = 'b')
plt.bar(ind, W[:15,2], color = 'g')
plt.ylim((0,1))
topic_labels = ['Topic #{}'.format(k) for k in range(K)]
plt.legend(topic_labels  + term_ranking, loc = 0)
plt.show()

doc_rankings = []
k = W.shape[1]
for topic_index in range(k):
	w = np.array( W[:,topic_index] )
	top_indices = np.argsort(w)[::-1]
	doc_rankings.append(top_indices)
print doc_rankings

text_xs = [0, 252, 433, 965, 1527, 1984, 2601, 3308, 3584, 3786, 4095, 4350, 4574, 4870, 5107, 5440, 5965]#np.array(volume_indexes) + np.diff(np.array(volume_indexes + [max(xs)]))/2
f = 0
while f in range(16):
    print (text_xs[f:f+2])
    f = f + 1