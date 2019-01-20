# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 17:18:50 2016

@author: user
"""

import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
import re
import codecs
import seaborn as sns
import scipy
from collections import Counter
import itertools
#Rename multiple files
path = ('Documents/Data/ICCS_2016')
path = ('D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\ICCS by year\\ICCS_2017_text')
files = os.listdir(path)
order = 000
for file in files:
    os.rename(os.path.join(path,file), os.path.join(path,'ICCS_2017_' + str(order)))
    order = order + 1
#
dir_data = "D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Data"
filenames = sorted([os.path.join(dir_data,fn) for fn in os.listdir(dir_data)])
stopwords = open("D:\\Topic Modeling\\stop words.txt", 'r')
stopwords = stopwords.read()
stopwords = stopwords.split()
doc_id = sorted(os.listdir(dir_data))

###################################
#file_paths = [os.path.join(dir_data, fname) for fname in os.listdir(dir_data) if fname.endswith(".txt") ]
#documents = [codecs.open(file_path, 'r', encoding="utf8").read() for file_path in file_paths ]
#Stanford NER
from nltk.tag.stanford import StanfordNERTagger
java_path = "C:/Program Files/Java/jdk1.8.0_144/bin/java.exe"
os.environ['JAVAHOME'] = java_path
st = StanfordNERTagger('D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\JoCS\\stanford-ner\\english.all.3class.distsim.crf.ser.gz',
                       'D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\JoCS\\stanford-ner\\stanford-ner-2017-06-09\\stanford-ner.jar')

def named_entity_recofnizor(test):
    ner = []
    for sent in nltk.sent_tokenize(test):
        tokens = nltk.tokenize.word_tokenize(sent)
        tags = st.tag(tokens)
        for tag in tags:
            if tag[1] in ["PERSON", "LOCATION", "ORGANIZATION"]:
                ner.append(tag[0])                
        return ner
                #print (tag[0])
                
#list of names
path = ('D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Data')
files = os.listdir(path)
names = []
for file in files:
    data = open(os.path.join(path,file), 'r', encoding = 'utf8')
    data = data.read()
    res = named_entity_recofnizor(data)
    names = names + res
    
list_of_names = pd.DataFrame(names, columns = ['list_of_names'])
name_list = list_of_names.drop_duplicates()
name_list.to_csv('D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\JoCS\\list_of_names.csv')
list_of_names.to_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\JoCS\\list_of_names_full.csv")
name_list = open("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\JoCS\\names.txt", 'r')
name_list = name_list.read()
name_list = name_list.split()
####  
token_pattern = re.compile(r"\b\w\w+\b", re.U)
stemmer = PorterStemmer()
topic_terms = []
doc_rankings = []             
def custom_tokenizer(s, min_term_length = 2 , max_term_length = 30):
    #named = named_entity_recofnizor(s)
    #list_of_names.append(named)
    res = [x.lower() for x in token_pattern.findall(s) if (x not in name_list and len(x) > min_term_length and len(x) <=max_term_length and x[0].isalpha() ) ]
    res = [token for token in res if token not in stopwords]
    res2 = [stemmer.stem(plural) for plural in res]
    res3 = [token for token in res2 if token not in stopwords]
    return [token for token in res3 if re.match(r'[A-Z]+', token, re.I)]

def preprocess( filenames, tokenizer=custom_tokenizer):
    tfidf = TfidfVectorizer(input = 'filename', stop_words='english', tokenizer=tokenizer, analyzer='word', token_pattern='(?u)\b\w\w+\b', max_df= 1.0, min_df= 0.0, lowercase=True, strip_accents="unicode", use_idf=True, norm="l2", ngram_range = (1,1)) 
    A = tfidf.fit_transform(filenames)
    terms = []
	
    num_terms = len(tfidf.vocabulary_)
    terms = [""] * num_terms
    for term in tfidf.vocabulary_.keys():
        terms[ tfidf.vocabulary_[term] ] = term
    #
    model = decomposition.NMF(init="nndsvd", n_components=100, max_iter=800)
    W = model.fit_transform(A)
    H = model.components_
    #
    
    for topic_index in range( H.shape[0] ):
        top_indices = np.argsort( H[topic_index,:] )[::-1][0:10]
        term_ranking = [terms[i] for i in top_indices]
        topic_terms.append(term_ranking)
        print ("Topic %d: %s" % ( topic_index, ", ".join( term_ranking ) ))
    
    volume_names = ['01', '02', '03', '04', '05', '06', '07', '08', '09','10', '11', '12', '13', '14', '15', '16', '17']
    volume_indexes = []
    for volname in volume_names:
        for i, paper_name in enumerate(filenames):
            if volname in paper_name:
                volume_indexes.append(i)
                break
    series_smooth2 = []
    doc = []
    series_smooth3 = []
    for i in range(100):
        #plt.clf()  # clears the current plot
        #plt.figure(figsize = (18.5, 10.5))
        
        series = W[:, i]  
        xs = np.arange(len(series))
        text_xs = [0, 251, 432, 964, 1526, 1983, 2600, 3307, 3583, 3785, 4094, 4349, 4573, 4869, 5106, 5439, 5695, 5981]#np.array(volume_indexes) + np.diff(np.array(volume_indexes + [max(xs)]))/2
        text_ys = np.repeat(max(series), len(volume_names)) - 0.05
        #n = [253, 184, 532, 562, 457, 617, 707, 276, 202, 309, 255, 224, 296, 237, 333, 255]
        series_smooth = pd.rolling_mean(series, 100)
        series_smooth2.append([])
        series_smooth3.append([])
        f = 0
        while f in range(17):
            data = series[text_xs[f]:text_xs[f+1]]
            aver = []
            for z in data:
                if z > 0:
                    aver.append(z)      
            series_smooth3[i].append(np.mean(aver))
            series_smooth2[i].append(np.mean(series[text_xs[f]:text_xs[f+1]]))
            f = f + 1
        '''p1 = plt.plot(series, '.')
        p2 = plt.plot(series_smooth2, '-', linewidth=2)
        plt.vlines(text_xs, ymin=0, ymax=np.max(series))
        for x, y, s in zip(text_xs, text_ys, volume_names):
            plt.text(x, y, s)
        plt.title("Topic {}: {}".format(i, topic_terms[i]))
        plt.ylabel("Topic share(0-1)")
        plt.xlabel("ICCS papers segemented per year")
        plt.ylim(0, max(series))
        plt.xlim(0, 5965)
        plt.legend((p1[0],p2[0]), ('topic share', 'moving average'))
        savefig_fn = "D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Plots\\Final\\ICCS-stem-K100-topic{}.png".format(i)
        plt.savefig(savefig_fn, format='png')'''
    MA = pd.DataFrame(series_smooth2).T
    MA.to_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\JoCS\\MovingAvg_before domain specific stopwords.csv")
    #MA2 = pd.DataFrame(series_smooth3)
    #MA2.to_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Plots\\Final\\MAFinalave100.csv")
    TTT = pd.DataFrame(topic_terms)
    TTT.to_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\JoCS\\TopicTerms_before domain specific stopwords.csv")
    WC = pd.DataFrame(W)
    WC.to_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\JoCS\\Papers_Contribution_before domain specific stopwords.csv")
    HC = pd.DataFrame(H).T
    HC.to_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\JoCS\\HC_before domain specific stopwords.csv")
    
    k = W.shape[1]
    for topic_index in range(k):
        w = np.array( W[:,topic_index] )
        top_indices = np.argsort(w)[::-1]
        doc_rankings.append(top_indices)
        
    #base_name = os.path.splitext( os.path.split(filenames)[-1] )[0]
    actual_top = min(50,len(filenames))
    #doc_path = "D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Plots"
    #log.info("Writing top %d document IDs to %s" % (options.top,doc_path) )
    fout = codecs.open("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\JoCS\\doc_topic.csv", "w", "utf-8")
    fout.write("Rank")
    for i in range(100):
        fout.write(",Topic %s" % i)
    fout.write("\n")
    for pos in range(actual_top):
        fout.write( "%d" % (pos + 1) )
        for ranking in doc_rankings:
            fout.write( ",%s" % doc_id[ranking[pos]])
        fout.write("\n")
    fout.close()
    return;
preprocess(filenames)


WC = pd.read_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\JoCS\\Final\Papers_Contribution.csv")
col_list = list(WC.columns[1:])
plt.figure(figsize = (14, 6))
for i in col_list[1:]:
    plt.plot(np.sort(WC[i]))
plt.show()
res = []
t = 0
for i in col_list:
    res.append([])
    for s, r in zip(WC[i], WC['Rank']):
        if s > 0.025:
            res[t].append(doc_id[r])
    t = t + 1
res = pd.DataFrame(res).T      
res.columns = col_list
res.to_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\JoCS\\Final\\docs_in_topic_075.csv")                
year = ['2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009','2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']

#data = pd.read_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Plots\\Final\\Average topic weights_T_Final.csv")
#col_list = list(Data.columns)
#core = []
papers = []
Data = res
DNA = {}
for k in col_list:
    for p in Data[k]:
        for c in col_list:
            if type(p) == str:
                if k != c:
                    if p in list(Data[c]) and p[5:9] == year[16]:
                        papers.append(p)
                        res = str(k + "|" + c)
                        res1 = str(c + "|" + k)
                        if res in DNA:
                            DNA[res] = DNA[res] + 1
                        elif res1 in DNA:
                            res = res1
                            DNA[res] = DNA[res]
                        elif res not in DNA:
                            DNA.update({res:1})
                        elif res1 not in DNA:
                            res = res1
                            DNA.update({res:1})
                            
DNA1 = []
for k,v in zip(DNA.keys(),DNA.values()):
    DNA1.append([k,v])
            
DNA1 = pd.DataFrame(DNA1, columns = ['Topics', 'Weight'])
DNA1.to_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\JoCS\\Final\\DNA_025_2001.csv")        



col = list(Data.columns)[1:]
plt.figure(figsize = (18.5, 10.5))
for i in col:
    plt.plot(Data.Papers, sorted(Data[i], reverse = True))
topic_labels = [format(i) for i in col[:50]]
topic_labels2 = [format(i) for i in col[50:]]
#plt.legend(topic_labels, loc = 0, fontsize = 14)
plt.title('Papers contribution to topic', fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)


Data = pd.read_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Plots\\Final\\WC1005.csv")
data = pd.read_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Plots\\Final\\Average topic weights_T_Final.csv")
col_list = list(Data.columns)
DNA = {}
for k in col_list:
    for p in Data[k]:
        for c in col_list:
            if type(p) == str:
                if k != c:
                    if p in list(Data[c]):
                        co = scipy.stats.pearsonr(data[k],data[c])
                        res = str(k + "|" + c + "|" + str(co))
                        res1 = str(c + "|" + k + "|" + str(co))
                        if res in DNA:
                            DNA[res] = DNA[res] + 1
                        elif res1 in DNA:
                            res = res1
                            DNA[res] = DNA[res]
                        elif res not in DNA:
                            DNA.update({res:1})
                        elif res1 not in DNA:
                            res = res1
                            DNA.update({res:1})
                            
DNA1 = []
for k,v in zip(DNA.keys(),DNA.values()):
    DNA1.append([k,v])
            
DNA1 = pd.DataFrame(DNA1)
DNA1.to_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Plots\\Final\\ICCS_Topics_correlation.csv")



Data = pd.read_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Plots\\Final\\WC1005.csv")
no_papers = []
for i in col_list:
    res = []
    for p in Data[i]:
        if type(p) == str:
            res.append(p)
    #res = list(Data[i])
    #print (len(res))
    no_papers.append([i, len(res)])
No_papers = pd.DataFrame(no_papers)
No_papers.to_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Plots\\Final\\No of papers in a topic5.csv")


Average_Variance = pd.read_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Plots\\Final\\Average topic variance.csv")
columns = list(Average_Variance.columns)
columns.remove('Mean')
plt.figure(figsize=(8, 6))
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('Year', fontsize=13)
plt.ylabel('Normalized topic proportion', fontsize=13)
plt.xlim(2000.5, 2016.5)
markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
color_list = ["orange", "yellow", "red", "green", "blue", "cyan", "violet", "black", 'grey']
c = 0
for i in columns:
    plt.plot(Average_Variance[i], marker = markers[c] , color =color_list[c], lw = 3)
    c = c + 1
plt.legend(fontsize=13)
plt.title('Highly variable high level ICCS topics', fontsize=13)
plt.show()

Plot = pd.read_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\JoCS\\Final\\Results\\Topic Terms_Labeled_for Plot.csv")