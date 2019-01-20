
# coding: utf-8

# In[2]:

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pandas as pd
import seaborn as sns
import scipy as sp
sns.set_style("whitegrid")


# In[3]:

topic = pd.read_csv('D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\JoCS\\Final\\Results\\Average Weight of Topics T.csv')
topic.sort('High Level').head()


# In[3]:

Data = pd.read_csv('D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Plots\\Final\\Topic DNA\\DN of ICCS Topics\\Average topic weights with groupings.csv')
Data.head()


# In[4]:

H_group_scores =  topic.groupby("High Level").sum()
topic[["Topic Modeling Results", "High Level"]].groupby("High Level").count()


# In[5]:

H_group_scores2 =  Data.groupby("High Level").sum()
Data[["Topics", "High Level"]].groupby("High Level").count()


# In[6]:

H_data_cols = H_group_scores.columns
# Normalize
H_scores_sum = H_group_scores.sum()
for dc_name in H_data_cols:
    H_group_scores[dc_name] = H_group_scores[dc_name] / H_scores_sum[dc_name]
H_group_scores.columns


# In[7]:

H_data_cols2 = H_group_scores2.columns
# Normalize
H_scores_sum2 = H_group_scores2.sum()
for dc_name in H_data_cols2:
    H_group_scores2[dc_name] = H_group_scores2[dc_name] / H_scores_sum2[dc_name]
H_group_scores2.columns


# In[4]:

H_group_scores['2001'].values/H_group_scores['2001'].sum() * 100, H_group_scores['2001'].index


# In[5]:

H_group_scores_percent = pd.DataFrame([H_group_scores[i].values/H_group_scores[i].sum() * 100 for i in H_group_scores.columns]
                                     )


# In[6]:

H_group_scores_percent.columns = H_group_scores['2001'].index
H_group_scores_percent.index = H_group_scores.columns


# In[8]:

H_group_scores_percent.to_csv('D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\JoCS\\Final\\Results\\Average Weight of Topics in percent.csv')


# In[13]:

H_group_scores = H_group_scores.T
#H_group_scores.index = H_group_scores[H_group_scores.columns[0]]
H_group_scores.head()


# In[14]:

H_group_scores.columns


# In[9]:

H_group_scores2 = H_group_scores2.T
H_group_scores2.head()


# In[8]:

from pandas.tools.plotting import autocorrelation_plot
plt.subplot(121), autocorrelation_plot(H_group_scores_percent["HPC"])


# In[107]:

H_group_scores_percent.columns


# In[10]:

#change point analysis
CU_SUM = []
Mean = []
t = 0
for c in H_group_scores.columns:
    mean = H_group_scores[c].mean()
    Mean.append(mean)
    CU_SUM.append([0])
    for i in range(len(H_group_scores[c])):
        s = CU_SUM[t][-1] + (H_group_scores[c][i] - mean)
        CU_SUM[t].append(s)
    t = t + 1


# In[22]:

year = ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009',
       '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
CU_SUM_pd = pd.DataFrame(CU_SUM)
CU_SUM_pd = CU_SUM_pd.T
CU_SUM_pd.columns = H_group_scores.columns
CU_SUM_pd.index = year


# In[10]:

year = ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009',
       '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
color_list = []
for name, hex in matplotlib.colors.cnames.items():
    color_list.append(name)
print(color_list[13:20])


# In[11]:

color = ['mediumpurple', 'palegoldenrod','mediumspringgreen', 'royalblue', 'lightslategrey', 'darkgoldenrod', 'lightseagreen', 'plum', 'orangered', 'sienna', 'greenyellow', 'darkviolet', 'darksage','forestgreen', 'fuchsia']
markers = ['.','o','v','^', '>','8','s','p','*','h','H','+','x','D','d','|','_']
year = [int(y) for y in year]
year[1:]


# In[83]:

plt.figure(figsize=(12, 8))
plt.rcParams['axes.facecolor'] = 'white'
plt.xticks(year, rotation = 'vertical', fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Raw data used for CUSUM', fontsize=20)
plt.xlim(2000.5,2017.5)
t = 0
for i in H_group_scores.columns:
    plt.plot( H_group_scores_percent[i].rolling(2).mean(), label = H_group_scores.columns[t], marker = markers[t], lw = 3, color = color[t], ms = 10.0)
    t = t + 1
plt.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize=20, ncol=1)
plt.title("High Level Topics' Proportion", fontsize = 20)
plt.grid(color='black', alpha=0.5, linestyle='dashed', linewidth=0.2)
plt.savefig('High Level ICCS Topics proportion raw data.png', format='png', dpi=300, bbox_inches="tight")
plt.show()


# In[17]:

legend = ['Modeling', 'HPC',  'Machine Learning', 'eScience', 'Programming', 'Optimization', 'Numerical', 'Visualisation',
          'Networks', 'Simulation ', 'Data', 'Security','Education']
#Middle_level_topics.index = year
#year = [int(i) for i in list(CU_SUM_pd.index)]
import matplotlib as mpl
mpl.style.use('classic')
plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.rcParams['axes.facecolor'] = 'white'
plt.xticks(year, rotation = 'vertical', fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Proportion in percentile', fontsize=20)
plt.text(2001, 24, 'a)', fontsize = 20, color = 'red')
plt.xlim(2000.5,2017.5)
t = 0
for i in legend:
    plt.plot(H_group_scores_percent[i], marker = markers[t], lw = 4, color = color[t], ms = 10.0)
    t = t + 1
#plt.legend(legend, loc='center left', bbox_to_anchor=(1,0.5), fontsize=20, ncol=1)
plt.title("Proportion of High Level Topics", fontsize = 20)
#plt.grid(color='black', alpha=0.5, linestyle='dashed', linewidth=0.2)
plt.subplot(122)
plt.rcParams['axes.facecolor'] = 'white'
plt.xticks(year, rotation = 'vertical', fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Moving Average', fontsize=20)
plt.xlim(2000.5,2017.5)
t = 0
for i in legend:
    plt.plot(H_group_scores_percent[i].rolling(2).mean(), marker = markers[t], lw = 4, color = color[t], ms = 10.0)
    t = t + 1
plt.legend(legend, loc='center left', bbox_to_anchor=(-1.2,-0.3), fontsize=20, ncol=4)
plt.title("Trend of High Level Topics' Proportion", fontsize = 20)
plt.text(2001, 24, 'b)', fontsize = 20, color = 'red')
#plt.grid(color='black', alpha=0.5, linestyle='dashed', linewidth=0.2)
plt.savefig('D:\Topic Modeling\preprocessed corpus\ICCS_Whole\Submit to Scientometric\Trend of Highlevel.png', format='png', dpi=300, bbox_inches="tight")


# In[ ]:

plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.rcParams['axes.facecolor'] = 'white'
plt.xticks(year, rotation = 'vertical', fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Year', fontsize=20)
plt.ylabel('In percent', fontsize=20)
plt.text(2001, 24, 'a)', fontsize = 20, color = 'red')
plt.xlim(2000.5,2017.5)
autocorrelation_plot(H_group_scores_percent["HPC"])
#plt.legend(legend, loc='center left', bbox_to_anchor=(1,0.5), fontsize=20, ncol=1)
plt.title("Proportion of High Level Topics", fontsize = 20)
#plt.grid(color='black', alpha=0.5, linestyle='dashed', linewidth=0.2)
plt.subplot(122)
plt.rcParams['axes.facecolor'] = 'white'
plt.xticks(year, rotation = 'vertical', fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Moving Average', fontsize=20)
plt.xlim(2000.5,2017.5)
autocorrelation_plot(H_group_scores_percent["Modeling"])
plt.legend(legend, loc='center left', bbox_to_anchor=(-1.2,-0.3), fontsize=20, ncol=4)
plt.title("Moving Average of High Level Topics' Proportion", fontsize = 20)
plt.text(2001, 0.057, 'b)', fontsize = 20, color = 'red')


# In[263]:

H_group_scores.plot.area(stacked = False, figsize = (14, 8), lw = 4)
plt.xticks( rotation = 'vertical', fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Year', fontsize=20)
#plt.ylim(0,1.01)
plt.ylabel('Proportion of Topics', fontsize=20)
plt.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize=20, ncol=1)
plt.grid(color='black', alpha=0.5, linestyle='dashed', linewidth=0.1)


# In[259]:

H_group_scores.index


# In[150]:

Middle_level_topics = pd.read_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\JoCS\\Final\\Results\\Highly variable middle level topics.csv")
legend = ['HPC', 'Modeling', 'Optimization', 'Numerical', 'eScience', 'Education', 'Data', 'Security', 
          'Simulation', 'Visualization', 'Programming', 'Machine Learining', 'Networks']
Middle_level_topics.index = year
plt.figure(figsize=(14, 8))
plt.rcParams['axes.facecolor'] = 'white'
plt.xticks(year, rotation = 'vertical', fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Year', fontsize=20)
plt.ylabel('CUSUM', fontsize=20)
plt.xlim(1999.5,2017.5)
t = 0
for i in CU_SUM_pd.columns:
    plt.plot(CU_SUM_pd[i], label = H_group_scores.columns[t], marker = markers[t], lw = 4, color = color[t], ms = 10.0)
    t = t + 1
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20, ncol=1)
plt.title('CUSUM Chart for High Level Topics', fontsize = 20)
plt.grid(color='black', alpha=0.5, linestyle='dashed', linewidth=0.2)
plt.savefig('CUSUM Chart for High Level ICCS Topics.png', format='png', dpi=300, bbox_inches="tight")


# In[214]:

legend = ['HPC', 'Modeling', 'Optimization', 'Numerical', 'eScience', 'Education', 'Data', 'Security', 
          'Simulation ', 'Visualisation', 'Programming', 'Machine Learning', 'Networks']
Middle_level_topics.index = year
year = [int(i) for i in list(CU_SUM_pd.index)]
plt.figure(figsize=(14, 8))
plt.rcParams['axes.facecolor'] = 'white'
plt.xticks(year, rotation = 'vertical', fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Year', fontsize=20)
plt.ylabel('CUSUM', fontsize=20)
plt.xlim(1999.5,2017.5)
t = 0
L = list(reversed(legend))
for i in L:
    plt.plot(year, CU_SUM_pd[i], marker = markers[t], lw = 4, color = color[t], ms = 10.0)
    t = t + 1
plt.legend(L, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20, ncol=1)
plt.title('CUSUM Chart for High Level Topics', fontsize = 20)
plt.grid(color='black', alpha=0.5, linestyle='dashed', linewidth=0.2)
plt.savefig('CUSUM Chart for High Level ICCS Topics.tiff', format='tiff', dpi=300, bbox_inches="tight")


# In[103]:

plt.figure(figsize=(14, 8))
plt.rcParams['axes.facecolor'] = 'white'
plt.xticks(year, rotation = 'vertical', fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Year', fontsize=20)
plt.ylabel('CUSUM', fontsize=20)
plt.xlim(1999.5,2017.5)
t = 0
for i in Middle_level_topics.columns:
    plt.plot(Middle_level_topics[i], label = Middle_level_topics.columns[t], marker = markers[t], lw = 4, color = color[t], ms = 10.0)
    t = t + 1
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20, ncol=1)
plt.title('CUSUM Chart for Highly Variable Middle Level Topics', fontsize = 20)
plt.grid(color='black', alpha=0.5, linestyle='dashed', linewidth=0.2)
plt.savefig('CUSUM Chart for Middle Level ICCS Topics.png', format='png', dpi=300, bbox_inches="tight")


# In[491]:

Data = pd.read_csv('D:\\Endalk\\AI6063.csv')
N = 5
ind = np.arange(N)


# In[492]:

Data['SiC %']


# In[24]:

plt.figure(figsize = (12,8)), plt.bar(ind, Data['Al6063 %'], color = 'green', label = 'Al6063', align='center' ),
plt.bar(ind, Data['SiC %'], color = 'yellow', label = 'Silicon', align='center'), 
plt.bar(ind, Data['Mg %'], color = 'red', label = 'Magnesium', align='center'), 
plt.ylabel('% of Mixtures/Hardness', fontsize = 14), plt.yticks(fontsize = 14), 
plt.xlabel('Experiments', fontsize = 14), plt.xticks(ind, ('E1', 'E2', 'E3', 'E4', 'E5'), fontsize = 14), 
plt.legend(fontsize = 14, loc = 'best'), 
plt.plot(Data['BHN'],label = 'BHN', marker = '>', color = 'mediumspringgreen', lw = 3),
plt.plot(Data['VH'], label = 'VH', marker = '>', color = 'blue', lw = 3),
plt.legend(loc='center left', bbox_to_anchor=(1,0.81), fontsize=14),
plt.savefig('Endalk.png', format='png', dpi=300, bbox_inches="tight")
plt.show()


# In[99]:

m = list(abs(CU_SUM_pd['Networks']))
m.index(np.max(m))
len(CU_SUM_pd.columns)


# In[125]:

for i in CU_SUM_pd.columns:
    m = list(abs(CU_SUM_pd[i]))
    s = np.sort(m)
    print ( m.index(np.max(m)) + 1,  m.index(s[-2]) +1)


# In[79]:

plt.plot(CU_SUM[3]), plt.title(Mean[3])


# In[18]:

estimator_magnitude_change = []
for topic in CU_SUM:
    estimator_magnitude_change.append(np.max(topic) - np.min(topic))
estimator_magnitude_change


# In[19]:

bootstrap = []
t = 0
for i in CU_SUM:
    bootstrap.append([])
    for b in range(1000):
        bootstrap[t].append(np.random.choice(i, size = 18))
    t = t + 1


# In[20]:

bootstrap[0][0]
np.mean(bootstrap[0][0])


# In[21]:

Boot_CU_SUM = []
Boot_Mean = []
t = 0
for c in range(len(bootstrap)):
    Boot_CU_SUM.append([])
    Boot_Mean.append([])
    for b in range(1000):
        mean = np.mean(bootstrap[c][b])
        Boot_Mean[t].append(mean)
        Boot_CU_SUM[t].append([0])
        for i in range(len(bootstrap[c][b])):
            s = Boot_CU_SUM[t][b][-1] + (bootstrap[c][b][i] - mean)
            Boot_CU_SUM[t][b].append(s)
    t = t + 1


# In[22]:

len(Boot_CU_SUM[0])


# In[23]:

boot_estimator_magnitude_change = []
conf_inter = []
t = 0
for i in range(len(Boot_CU_SUM)):
    conf_inter.append([])
    boot_estimator_magnitude_change.append([])
    for topic in Boot_CU_SUM[i]:
        boot_dif = np.max(topic) - np.min(topic)
        boot_estimator_magnitude_change[t].append(boot_dif)
        conf_inter[t].append(boot_dif < estimator_magnitude_change[i])
    t = t + 1


# In[24]:

Conf_Inter = pd.DataFrame(conf_inter)
CU_SUM_pd = pd.DataFrame(CU_SUM)
CU_SUM_pd = CU_SUM_pd.T
CU_SUM_pd.columns = H_group_scores.columns
CU_SUM_pd.index = year
CU_SUM_pd.head()


# In[25]:

t = 0
for i in H_group_scores.columns:
    CI = pd.DataFrame(conf_inter[t], columns = ['Bootstrap'])
    print ('CI ' + i + ' = ', str(CI.Bootstrap.value_counts()[0]/1000*100))
    t = t + 1


# In[26]:

Boot_estimator_magnitude_change = []
for topic in Boot_CU_SUM:
    Boot_estimator_magnitude_change.append(np.max(topic) - np.min(topic))


# In[27]:

#plt.xlim(2000, 2017)
plt.plot(CU_SUM[2], label = 'Original Data', lw = 3),
for i in range(len(Boot_CU_SUM[2][:15])):
    plt.plot(Boot_CU_SUM[2][i], label = str (i + 1) + '_Bootstrap', marker = markers[i], lw = 3, color = color_list[i], ms = 10.0)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
plt.show()


# In[28]:

plt.figure(figsize=(14, 8))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('CUMSUM', fontsize=14)
plt.xlim(2000,2018)
markers = ['.','o','v','^', '<', '>', '1', '2','3','4','8','s','p','*','h','H','+','x','D','d','|','_']
t = 0
for i in H_group_scores.columns:
    CI = pd.DataFrame(conf_inter[t], columns = ['Bootstrap'])
    if round(CI.Bootstrap.value_counts()[0]/1000*100) > 80:
        #print ('CI ' + i + ' = ', str(round(CI.Bootstrap.value_counts()[0]/1000*100)))
        plt.plot(CU_SUM_pd[i], label = i + ' with ' + str(round(CI.Bootstrap.value_counts()[0]/1000*100)) + '% CI', lw = 3, marker = markers[t], color = color[t], ms = 10.0)
    t = t + 1
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
plt.title('CUSUM chart for Highly Variable High Level ICCS Topics', fontsize = 14)
plt.show()


# In[29]:

#Bootstrtap
plt.figure(figsize=(12, 8))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('CUMSUM', fontsize=14)
markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
#plt.xlim(2000, 2017)
#color_list = ["orange", "yellow", "red", "green", "blue", "cyan", "violet", "black", 'grey']
t = 0
for topic in Boot_CU_SUM:
    for boot in topic:
        plt.plot(boot, label = H_group_scores.columns[t], marker = markers[t], lw = 3)
    t = t + 1
#plt.legend(fontsize=14)
plt.title('Boot_CUSUM chart for High Level Topics', fontsize = 14)
plt.show()


# In[30]:

plt.plot(CU_SUM[0]), plt.plot(Boot_CU_SUM[0])


# In[70]:

plt.figure(figsize=(12, 8))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Topic score', fontsize=14)
markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
plt.xlim(2000.5, 2016.5)
m = 0
for i in H_group_scores.columns:
    plt.plot(H_group_scores[i], marker = markers[m])
    m = m + 1
plt.legend(fontsize=14)
plt.show()


# In[31]:

H_N = len(H_group_scores.columns)
H_slp = [sp.stats.linregress(np.arange(H_N), np.array(H_group_scores.iloc[[i]][H_group_scores.columns]).flatten()).slope for i in range(len(H_group_scores.columns))]
sns.distplot(H_slp)


# In[32]:

len(H_slp)


# In[33]:

H_variat = np.std(np.array(H_group_scores[H_group_scores.columns]), axis = 1) / np.average(np.array(H_group_scores[H_group_scores.columns]), axis = 1)
sns.distplot(H_variat)


# In[34]:

sns.plt.plot(H_variat)
sns.plt.title(np.mean(H_variat))


# In[77]:

H_filter = np.abs(H_slp) > 0.002
H_res_dy = H_group_scores.columns[H_filter]
H_res_dy


# In[76]:

H_group_scores[H_res_dy]


# In[37]:

Average_Variance = H_group_scores[H_res_dy]
#Average_Variance = Average_Variance(["Average Variance"])
Average_Variance.iloc[[0]]
#Average_Variance['Data'].mean()
Average_Variance.to_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\JoCS\\Final\\Results\\topic variance.csv")


# In[38]:

Average_Variance = pd.read_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\JoCS\\Final\\Results\\topic variance.csv")
Average_Variance.head()


# In[39]:

t_n = H_filter.sum()
plt.figure(figsize=(8, 7))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Normalized topic proportion', fontsize=14)
plt.xlim(2000.5, 2017.5)
color_list = ["orange", "yellow", "red", "green", "blue", "cyan", "violet", "black", 'grey']
c = 0
for i in H_res_dy:
    plt.plot(H_group_scores[i], marker = markers[c] , markersize=10, color =color_list[c], lw = 2)
    c = c + 1
plt.legend(fontsize=14)
plt.title('Highly variable high level ICCS topics', fontsize=14)
plt.show()


# In[50]:

columns = list(Average_Variance.columns)
#columns.remove('Mean')
plt.figure(figsize=(8, 6))
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('Year', fontsize=13)
plt.ylabel('Normalized topic proportion', fontsize=13)
plt.xlim(2000.5, 2017.5)
markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
color_list = ["orange", "yellow", "red", "green", "blue", "cyan", "violet", "black", 'grey']
c = 0
for i in columns:
    plt.plot(Average_Variance[i], marker = markers[c] , color =color_list[c], lw = 3)
    c = c + 1
plt.legend(fontsize=13)
plt.title('Highly variable high level ICCS topics', fontsize=13)
plt.show()


# In[48]:

list(Average_Variance.columns)


# In[37]:

plt.plot(Average_Variance[columns[8]])


# In[38]:

plt.figure(figsize=(12, 8))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Topic score', fontsize=14)
plt.xlim(2000.5, 2016.5)
c = 0
for i in H_group_scores.columns:
    if i not in H_res_dy:
        plt.plot(H_group_scores[i], marker = markers[c], color = color_list[c], lw = 3)
        c = c + 1
plt.legend(fontsize=14)
plt.title('Less trendy high level ICCS topics', fontsize=14)
plt.show()


# In[40]:

list_H = ['Data', 'Machine Learning','Networks', 'Programming','Simulation ', 'Visualisation', 'eScience']
for i in list_H:
    plt.plot(H_group_scores[i])
plt.legend(fontsize=14)
plt.show()


# In[89]:

from collections import OrderedDict
plt.figure(figsize=(14, 6))
plt.subplot(121)
res = {}
for i in H_group_scores.columns:
    res.update({i:H_group_scores[i].sum()})
d_sorted_by_value = OrderedDict(sorted(res.items(), key=lambda x: x[1]))
plt.rcParams['axes.facecolor'] = 'white'
plt.xticks(fontsize=14, color = 'black')
plt.yticks(fontsize=14, color = 'black')
plt.xlabel('Proportion of Topic', fontsize = 14)
plt.xlim(0.0,3.7)
objects = list(d_sorted_by_value.keys())
y_pos = np.arange(len(objects))    
res1 = list(d_sorted_by_value.values())
plt.barh(y_pos, res1,color = 'green', align = 'center', lw = 2)
plt.yticks(y_pos, objects)
plt.title('High level ICCS topics\' proportion from 2001-2017', fontsize=14, color = 'black')
plt.grid(color='black', alpha=0.5, linestyle='dashed', linewidth=0.2)
plt.subplot(122)
res2 = {}
for i in H_group_scores2.columns:
    res2.update({i:H_group_scores2[i].sum()})
d_sorted_by_value = OrderedDict(sorted(res2.items(), key=lambda x: x[1]))
plt.rcParams['axes.facecolor'] = 'white'
plt.xticks(fontsize=14, color = 'black')
plt.yticks(fontsize=14, color = 'black')
plt.xlabel('Proportion of Topic', fontsize = 14)
plt.xlim(0.0,3.7)
objects = list(d_sorted_by_value.keys())
y_pos = np.arange(len(objects))    
res1 = list(d_sorted_by_value.values())
plt.barh(y_pos, res1,color = 'blue', align = 'center', lw = 2)
plt.yticks(y_pos, objects)
plt.tight_layout()
plt.title('High level ICCS topics\' proportion from 2001-2016', fontsize=14, color = 'black')
plt.grid(color='black', alpha=0.5, linestyle='dashed', linewidth=0.2)
plt.savefig('Comparision of High level ICCS topics share.png', format='png', dpi=300, bbox_inches="tight")
plt.show()


# In[53]:

res


# In[93]:

plt.figure(figsize = (14,8))
plt.rcParams['font.size'] = 14
plt.subplot(121)
order_pie = OrderedDict(sorted(res.items(), reverse=True, key=lambda x: x[1]))
labels = list(order_pie.keys())
sizes = list(order_pie.values())

#labels = list(res.keys())
#sizes = list(res.values())
colors = ['mediumpurple', 'palegoldenrod','mediumspringgreen', 'royalblue', 'lightslategrey', 'darkgoldenrod', 'lightseagreen', 'plum', 'orangered', 'sienna', 'greenyellow', 'darkviolet', 'darksage','forestgreen', 'fuchsia']
explode = (0.1, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0)  # explode 1st slice
plt.pie(sizes, colors=colors, shadow=False, autopct='%1.1f%%', startangle=140, explode = explode)
plt.axis('equal')
plt.legend(labels, loc='best', bbox_to_anchor=(1, 0.9), fontsize=20, ncol = 1)
plt.title('High Level ICCS Topics\' Proportion in all Dataset', fontsize = 20)
plt.tight_layout()
#plt.grid(color='black', alpha=0.5, linestyle='dashed', linewidth=0.2)
plt.savefig('ICCS Topics share in the all dataset.png', format='png', dpi=300, bbox_inches="tight")


# In[223]:

pie_list2001 = {}
pie_list2009 = {}
pie_list2017 = {}
pie_list = {}


# In[224]:

Data = H_group_scores
t = 0
for c, y in zip(Data.columns, year[1:]):
    pie_list.update({c:list(Data.loc[[y]][c])[0]})
    


# In[218]:

H_group_scores.head()


# In[221]:

year[1:]


# In[220]:

data = Data.T
listper = []
for y in year[1:]:
    dic = {}
    for v, k in zip(data[y],data.index):
        dic.update({k:v/data[y].sum()*100})
    listper.append([dic])
    #listper.append([{k:v/data[y].sum()*100} for v, k in zip(data[y],data.index)])
listper[0][0]


# In[106]:

listtopic = []
for i in range (len(listper)):
    listtopic.append(list(listper[i][0].values()))
listtopic[0]    


# In[163]:

dataframe = pd.read_csv('D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\JoCS\\Final\\Results\\Evolution of High Level ICCS Topics Proportion.csv')


# In[120]:

dataframe.columns


# In[164]:

Da = dataframe.T


# In[182]:

Da_new = pd.read_csv('D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\JoCS\\Final\\Results\\da.csv')


# In[184]:

Da_new.columns


# In[193]:

Share_Sort = Da_new.sort_values(by=['2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008',
       '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'], ascending = False)


# In[196]:

Share_Sort.T.to_csv('D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\JoCS\\Final\\Results\\share.csv')


# In[78]:

share_topic = pd.read_csv('D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\JoCS\\Final\\Results\\share.csv')
share_topic.columns


# In[79]:

share_topic2 = share_topic[['HPC', 'Modeling', 'Machine Learning', 'eScience',
       'Visualisation', 'Programming', 'Optimization', 'Numerical', 'Networks',
       'Simulation ', 'Data', 'Security', 'Education']]
share_topic2.index = share_topic.Year


# In[213]:

share_topic2.plot(kind = 'bar', stacked = True)


# In[80]:

sorted_proportion = []
for i in range(17):
    sorted_proportion.append(sorted(dataframe2.iloc[i], reverse = True))


# In[159]:

sorted_proportion_pd = pd.DataFrame(sorted_proportion, columns = dataframe.columns[1:] )


# In[160]:

sorted_proportion_pd.plot(kind = 'bar', stacked = True)


# In[75]:

dataframe.to_csv('D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\JoCS\\Final\\Results\\Evolution of High Level ICCS Topics Proportion.csv')


# In[87]:

#plt.figure(figsize = (14,8)),
share_topic2.plot(kind='bar', stacked=True, color = color, width = 0.8, figsize=(12, 8))
plt.ylabel('Proportion of Topics in %', fontsize = 20), 
plt.yticks(fontsize = 20),
plt.ylim(0,100.5) 
plt.xlabel('Year', fontsize = 20), 
plt.xticks(fontsize = 20),
plt.title("Evolution of High Level ICCS Topics\' Proportion from 2001-2017", fontsize=20)
plt.grid(color='black', alpha=0.5, linestyle='dashed', linewidth=0.2)
plt.tight_layout()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20, ncol = 1)
plt.savefig('Evoluation of topics share over 17 years.eps', format='eps', dpi=300, bbox_inches="tight")


# In[114]:

share_topic2.plot(color = color, lw = 3, figsize=(12, 8))
plt.ylabel('Proportion of Topics in %', fontsize = 20), 
plt.yticks(fontsize = 20),
#plt.ylim(0,100.5) 
plt.xlabel('Year', fontsize = 20), 
plt.xticks(fontsize = 20),
plt.title("High Level ICCS Topics\' Proportion from 2001-2017", fontsize=20)
plt.grid(color='black', alpha=0.5, linestyle='dashed', linewidth=0.2)
plt.tight_layout()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20, ncol = 1)
plt.savefig('High Level ICCS Topics line plot.png', format='png', dpi=300, bbox_inches="tight")


# In[217]:

share_topic2.T.to_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Submit to Scientometric\\Topic Prediction in percent.csv")


# In[373]:

inputList = list(pie_list2001.values())
for k, v in zip(list(pie_list2001.keys()), list(pie_list2001.values())):
    print(k, round(v/np.sum(list(pie_list2001.values()))*100,1), '%')


# In[309]:

pie_list2001, pie_list2009, pie_list2017


# In[357]:

plt.figure(figsize = (12,6))
fig = plt.gcf()
fig.suptitle("Evolution of High Level ICCS Topic Proportion(2001, 2009 and 2017)", fontsize=14)
#plt.tight_layout()
fig.subplots_adjust(top=0.88)
plt.subplot(131)
pie1 = list(pie_list2001.values())
plt.rcParams['font.size'] = 14
plt.pie(pie1, colors=colors, shadow=True, autopct='%1.1f%%', startangle=140, explode = explode)
#plt.title('2001', fontsize = 14)
plt.subplot(132)
pie2 = list(pie_list2009.values())
plt.rcParams['font.size'] = 14
plt.pie(pie2, colors=colors, shadow=True, autopct='%1.1f%%', startangle=140, explode = explode)
#plt.title('2009', fontsize = 14)
plt.subplot(133)
pie3 = list(pie_list2017.values())
plt.rcParams['font.size'] = 14
plt.pie(pie3, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140, explode = explode)
#plt.title('2017', fontsize = 14)
plt.tight_layout()
plt.legend(list(pie_list2001.keys()), loc = 'lower center', bbox_to_anchor = (0,-0.12,1,1),
            bbox_transform = plt.gcf().transFigure, ncol = 4, fontsize = 14)
plt.savefig('comparing ICCS Topics share 2001, 2009 & 2017.png', format='png', dpi=300, bbox_inches="tight")
plt.show()


# In[274]:

Dgree_dist= pd.read_csv('D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Plots\\Final\\Topic DNA\\Final result of network of ICCS topics.csv')


# In[284]:

plt.figure(figsize=(12, 8))
plt.xticks(fontsize=14, color = 'black')
plt.yticks(fontsize=14, color = 'black')
plt.hist(Dgree_dist.degree, color = 'blue', lw = 2)
plt.xlim([0,56])
plt.title('Degree distribution of networks of low level ICCS topics', fontsize=14, color = 'black')
plt.xlabel("Number of neighbors", fontsize = 14, color = 'black')
plt.show()

