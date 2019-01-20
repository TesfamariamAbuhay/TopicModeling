
# coding: utf-8

# In[1]:

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


# In[2]:

topic = pd.read_csv('D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\JoCS\\Final\\Results\\Average Weight of Topics T.csv')
topic.head()


# In[6]:

group_scores =  topic.groupby("Middle Level").sum()
topic[["Topic Modeling Results", "Middle Level"]].groupby("Middle Level").count()


# In[7]:

data_cols = group_scores.columns
# Normalize
scores_sum = group_scores.sum()
for dc_name in data_cols:
    group_scores[dc_name] = group_scores[dc_name] / scores_sum[dc_name]
group_scores.head()


# In[8]:

group_scores = group_scores.T
group_scores.head()


# In[9]:

group_scores.T.to_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Submit to Scientometric\\Topic Prediction2.csv")


# In[195]:

group_scores = group_scores
listper_M = []
for y in list(group_scores.columns):
    dic = {}
    for v, k in zip(group_scores[y],group_scores.index):
        dic.update({k:v/group_scores[y].sum()*100})
    listper_M.append([dic])
    #listper.append([{k:v/data[y].sum()*100} for v, k in zip(data[y],data.index)])
listper_M[59][0]


# In[179]:

len(listper_M)


# In[196]:

Middle_Level = pd.DataFrame(list(listper_M[16][0].keys()),columns = ['Topic'])
#Middle_Level.index = Middle_Level.Topic
for i in range(17):
    Middle_Level[year[i]] = pd.DataFrame(list(listper_M[i][0].values()))


# In[199]:

Middle_Level = Middle_Level.T
Middle_Level


# In[48]:

Middle_Level = pd.read_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Submit to Scientometric\\Topic Prediction.csv")


# In[49]:

Middle_Level_c = list(Middle_Level.columns)
Middle_Level_T = Middle_Level[Middle_Level_c[1:]]


# In[ ]:




# In[413]:

L = list(Middle_Level_T.columns)[:2]
L


# In[2]:

Topics = ['HPC_Parallel', 'Machine Learning_Methods', 'Modeling_Data-driven', 'eScience_Decision Support Systems', 
          'Optimization_Metaheuristic', 'Visualisation_Human Computer Interaction(GUI)', 'Numerical_Methods', 'Programming_Method', 
          'Networks_Theory', 'Data_Theory', 'Simulation _Crowd Dynamics', 'Security_Attack Detection', 'Education_Education']
color = ['mediumpurple', 'palegoldenrod','mediumspringgreen', 'royalblue', 'lightslategrey', 'darkgoldenrod', 'lightseagreen', 'plum', 'orangered', 'sienna', 'greenyellow', 'darkviolet', 'darksage','forestgreen', 'fuchsia', 'burlywood', 'slateblue', 'lightslategrey']
markers = ['o','v','^', '>','8','s','p','*','h','H','+','x','D','d','|','_']
#year = [int(i) for i in list(group_scores.index)]


# In[187]:

group_scores_M = group_scores[Topics]


# In[391]:

percent = []
for y in year:
    res = [float(i/group_scores.T[str(y)].sum()*100) for i in group_scores.T[str(y)]]
    percent.append(res)


# In[392]:

Middle = pd.DataFrame(percent, columns = list(group_scores.columns))
Middle.index = year


# In[373]:

Middle_Level = pd.DataFrame(percent, columns = Topics)
Middle_Level.index = year


# In[3]:

Middle = pd.read_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Submit to Scientometric\\Topic Prediction Final 60.csv")


# In[4]:

Middle[Topics].plot.area(stacked = False, color = color, figsize = (14,8), lw = 4)
year = [int(i) for i in list(Middle.index)]
plt.xticks(year, rotation = 'vertical', fontsize=20)
plt.yticks(fontsize = 20)
plt.xlabel('Year', fontsize = 20)
plt.xlim(2000.8, 2017.2)
plt.ylabel('Proportion of Topics', fontsize = 20)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20, ncol=1)
plt.title("Proportion of Topics", fontsize = 20)
plt.grid(color='black', alpha=0.5, linestyle='dashed', linewidth=0.1)
plt.savefig('Topics Proportion_Prediction.png', format='png', dpi=300, bbox_inches="tight")


# In[5]:

Middle[Topics]


# In[494]:

Autocorrelation = []
for i in Topics:
    dataframe = pd.concat([Middle[i], Middle[i].shift(-1)], axis=1)
    dataframe.columns = ['t', 't+1']
    Autocorrelation.append([i,dataframe.corr().values[0][1]])
    #dataframe.head()


# In[506]:

for i in range(len(Autocorrelation)):
    if Autocorrelation[i][1] > 0.5:
        print(Autocorrelation[i])


# In[13]:

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# In[8]:

HPC = ['HPC_Parallel', 'eScience_Decision Support Systems','Machine Learning_Multimedia','Modeling_Data-driven','Networks_Routing',
'Networks_TCP','Optimization_Metaheuristic','Programming_Method','Simulation _Cellular automata','Simulation _Lattic Boltmann']


# In[9]:

Pre_HPC = Middle[HPC]


# In[10]:

Pre_HPC


# In[11]:

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
    data: Sequence of observations as a list or NumPy array.
    n_in: Number of lag observations as input (X).
    n_out: Number of observations as output (y).
    dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
    Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# In[16]:

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# In[17]:

# transform data to be stationary
series = Pre_HPC
# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)


# In[19]:

diff_values.shape


# In[20]:

Df_diff = pd.concat([pd.DataFrame(i).T for i in diff_values])


# In[21]:

Df_diff.shape, Pre_HPC.shape


# In[14]:

values = Pre_HPC.values
# ensure all data is float
diff_values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(diff_values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1)


# In[15]:

# drop columns we don't want to predict
#reframed3.drop(reframed3.columns[11:], axis=1, inplace=True)
reframed.drop(reframed.columns[11:], axis=1, inplace=True)
reframed


# In[16]:

# split into train and test sets
values = reframed.values
n_train_year = 11
train = values[:n_train_year, :]
test = values[n_train_year:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# In[17]:

# design network
model = Sequential()
model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
model.add(LSTM(15, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(7))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])


# In[18]:

model.summary()


# In[ ]:

# fit network
history = model.fit(train_X, train_y, epochs=3000, batch_size=1, validation_data=(test_X, test_y), verbose=2, shuffle=False)


# In[ ]:

# evaluate the model
scores = model.evaluate(test_X, test_y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[796]:

# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# In[803]:

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


# In[560]:

Middle.shape


# In[485]:

series = Middle[Topics[0]]
groups = series
years = pd.DataFrame()
for name, group in groups:
    years[name.year] = group.values
years.boxplot()


# In[493]:

type(groups)


# In[455]:

from pandas.tools.plotting import lag_plot
lag_plot(Middle[Topics[7]], color = 'blue')


# In[425]:

plt.figure(figsize = (14, 8))
m = 0
for i in Topics:
    plt.plot(Middle[i].rolling(window = 2).mean(), lw = 5, color = color[m], marker = markers[m], ms = 15)
    m+=1
plt.xticks(year, rotation = 'vertical', fontsize=20)
plt.yticks(fontsize = 20)
plt.xlabel('Year', fontsize = 20)
plt.xlim(2001.5, 2017.5)
plt.ylabel('Proportion of Topics', fontsize = 20)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20, ncol=1)
plt.grid(color='black', alpha=0.5, linestyle='dashed', linewidth=0.1)
plt.title("Moving Average of Proportion of Topics", fontsize = 20)
plt.savefig('Topics Proportion Moving Average.png', format='png', dpi=300, bbox_inches="tight")


# In[423]:

data = Middle[Topics[:5]]
g = sns.PairGrid(data)
g.map(plt.scatter);


# In[343]:

Correlation = group_scores.corr()
Correlation [Topics[0]]    


# In[271]:

group_scores_M.T[year[1]]


# In[491]:

Middle_Level[HPC].to_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\Submit to Scientometric\\HPC Topics Proportion.csv")


# In[7]:

Correlation = group_scores.corr()
Correlation['Numerical_Solvers']


# In[8]:

core_list = []
t = 0
for i in Correlation.columns:
    core_list.append([])
    core = Correlation[i] > 0.5
    core_list[t].append(core[core == True])
    t = t + 1


# In[9]:

for i in range(len(Correlation.columns)):
    print(Correlation.columns[i])
    print(core_list[i])


# In[9]:

plt.figure(figsize = (12,6))
plt.subplot(121), plt.scatter(group_scores['Data_Assimilation'], group_scores['Numerical_Solvers'], s = 55),
plt.xticks(fontsize = 14), plt.yticks(fontsize = 14), plt.xlabel('Data_Assimilation', fontsize = 14),  plt.ylabel('Numerical_Solvers', fontsize = 14),
plt.subplot(122),plt.scatter(group_scores['Modeling_Data-driven'], group_scores['Programming_Method'], s = 55)
plt.tight_layout()
plt.xticks(fontsize = 14), plt.yticks(fontsize = 14), plt.xlabel('Modeling_Data-driven', fontsize = 14),  plt.ylabel('Programming_Method', fontsize = 14)


# In[10]:

Data_corr = []
for c in Correlation.columns:
    Data = Correlation[c]
    Data_corr.append([c, Data[Data < -0.8]])
Data_corr


# In[11]:

from pandas.tools.plotting import lag_plot
#lag_plot(group_scores['HPC_Cloud'], color = 'blue')
dataframe = pd.concat([group_scores['Modeling_Finance'], group_scores['Modeling_Finance'].shift(-1)], axis=1)
dataframe.columns = ['t', 't+1']
dataframe.corr()


# In[12]:

group_scores.columns


# In[71]:

color = ['mediumpurple', 'palegoldenrod','mediumspringgreen', 'royalblue', 'lightslategrey', 'darkgoldenrod', 'lightseagreen', 'plum', 'orangered', 'sienna', 'greenyellow', 'darkviolet', 'darksage','forestgreen', 'fuchsia', 'burlywood', 'slateblue', 'lightslategrey']
markers = ['.','o','v','^', '>','8','s','p','*','h','H','+','x','D','d','|','_']


# In[157]:

growing_topics = ['HPC_Cloud', 'HPC_GPU', 'HPC_Power Comsumption', 'Modeling_Environmental', 'Modeling_Health','Modeling_Healthcare', 'Modeling_Multi-scale',
                 'Modeling_Transportation','Optimization_Applications', 'Optimization_Metaheuristic','eScience_Scheduling','eScience_Workflows']
fading_topics = ['Data_Databases', 'HPC_Fault tolerant', 'Machine Learning_AI_genetic algorithm', 'Machine Learning_Multimedia',
                'Modeling_Bioinformatics', 'Modeling_Biomedicine', 'Modeling_Chemistry', 'Modeling_Geometry', 'Networks_Routing', 'Networks_TCP',
               'Networks_Theory', 'Networks_Wireless','Programming_Method', 'Security_Encryption','Simulation _Cellular automata',
                'eScience_Grid','eScience_Problem-solving environment']
year = [int(i) for i in group_scores.index]
plt.figure(figsize = (14,8))
c = 0
for i in growing_topics:
    res = pd.rolling_mean(group_scores[i], window = 2)
    plt.plot(year, res, marker = markers[c], lw = 4, color = color[c], ms = 15)
    c = c + 1
plt.xticks(year[1:], fontsize = 20, rotation = 'vertical'), plt.yticks(fontsize = 20)
plt.legend(loc='center left', bbox_to_anchor=(-0.2, -0.3), fontsize=14, ncol = 2)
plt.title('Growing ICCS topics', fontsize=20)
plt.xlim(2001.5,2017.5)
plt.xlabel('Year', fontsize = 20), plt.ylabel('Rolling mean', fontsize = 20)
plt.tight_layout()
plt.grid(color='black', alpha=0.5, linestyle='dashed', linewidth=0.2)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20, ncol = 1)
plt.savefig('Growing ICCS Topics.eps', format='eps', dpi=300, bbox_inches="tight")


# In[160]:

year = [int(i) for i in group_scores.index]
plt.figure(figsize = (14,8))
c = 0
for i in fading_topics:
    res = pd.rolling_mean(group_scores[i], window = 2)
    plt.plot(res, marker = markers[c], lw = 4, color = color[c], ms = 15)
    c = c + 1
plt.xticks(year[1:], rotation = 'vertical', fontsize = 20), plt.yticks(fontsize = 20)
plt.title('Disappearing ICCS topics', fontsize=20)
plt.xlabel('Year', fontsize = 20), plt.ylabel('Rolling mean', fontsize = 20)
plt.xlim(2001.5,2017.5)
plt.tight_layout()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20, ncol = 1)
plt.grid(color='black', alpha=0.5, linestyle='dashed', linewidth=0.2)
plt.savefig('Disappearing ICCS Topics.eps', format='eps', dpi=300, bbox_inches="tight")


# In[17]:

len(fading_topics), len(color), len(markers)


# In[8]:

year = ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009',
       '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']


# In[9]:

CU_SUM_Mid = []
Mean_Mid = []
t = 0
for c in group_scores.columns:
    mean = group_scores[c].mean()
    Mean_Mid.append(mean)
    CU_SUM_Mid.append([0])
    for i in range(len(group_scores[c])):
        s = CU_SUM_Mid[t][-1] + (group_scores[c][i] - mean)
        CU_SUM_Mid[t].append(s)
    t = t + 1


# In[10]:

plt.figure(figsize=(14, 8))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('CUMSUM', fontsize=14)
markers = ['.','o','v','^', '<', '>', '1', '2','3','4','8','s','p','*','h','H','+','x','D','d','|','_']
color_list = []
for name, hex in matplotlib.colors.cnames.items():
    color_list.append(name)
t = 0
for i in CU_SUM_Mid:
    plt.plot(i, label = group_scores.columns[t], lw = 3, ms = 10.0)
    t = t + 1
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
plt.title('CUSUM chart for Middle Level Topics', fontsize = 14)
plt.show()


# In[11]:

estimator_magnitude_change_mid = []
for topic in CU_SUM_Mid:
    estimator_magnitude_change_mid.append(np.max(topic) - np.min(topic))
len(estimator_magnitude_change_mid)


# In[12]:

bootstrap_mid = []
t = 0
for i in CU_SUM_Mid:
    bootstrap_mid.append([])
    for b in range(1000):
        bootstrap_mid[t].append(np.random.choice(i, size = 18))
    t = t + 1
len(bootstrap_mid[0][0])


# In[13]:

Boot_CU_SUM_Mid = []
Boot_Mean_Mid = []
t = 0
for c in range(len(bootstrap_mid)):
    Boot_CU_SUM_Mid.append([])
    Boot_Mean_Mid.append([])
    for b in range(1000):
        mean = np.mean(bootstrap_mid[c][b])
        Boot_Mean_Mid[t].append(mean)
        Boot_CU_SUM_Mid[t].append([0])
        for i in range(len(bootstrap_mid[c][b])):
            s = Boot_CU_SUM_Mid[t][b][-1] + (bootstrap_mid[c][b][i] - mean)
            Boot_CU_SUM_Mid[t][b].append(s)
    t = t + 1


# In[14]:

boot_estimator_magnitude_change_mid = []
conf_inter_mid = []
t = 0
for i in range(len(Boot_CU_SUM_Mid)):
    conf_inter_mid.append([])
    boot_estimator_magnitude_change_mid.append([])
    for topic in Boot_CU_SUM_Mid[i]:
        boot_dif_mid = np.max(topic) - np.min(topic)
        boot_estimator_magnitude_change_mid[t].append(boot_dif_mid)
        conf_inter_mid[t].append(boot_dif_mid < estimator_magnitude_change_mid[i])
    t = t + 1


# In[15]:

Conf_Inter = pd.DataFrame(conf_inter_mid)
CU_SUM_Mid_pd = pd.DataFrame(CU_SUM_Mid)
CU_SUM_Mid_pd = CU_SUM_Mid_pd.T
CU_SUM_Mid_pd.columns = group_scores.columns


# In[16]:

CU_SUM_Mid_pd.index = year
CU_SUM_Mid_pd.head()


# In[21]:

color = ['mediumpurple', 'palegoldenrod','mediumspringgreen', 'royalblue','black', 'blue', 'lightslategrey', 'darkgoldenrod', 'lightseagreen', 'plum', 'orangered', 'sienna', 'greenyellow', 'darkviolet', 'darksage','forestgreen', 'fuchsia']
markers = ['.','o','v','^', '>','8','s','p','*','h','H','+','x','D','d','|','_']
len(markers), len(color)


# In[20]:

len(group_scores.columns)


# In[21]:

plt.figure(figsize=(10, 6))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('CUMSUM', fontsize=14)
plt.xlim(2000,2017.5)
plt.ylim(-0.125,0.08)
t = 0
c = 0
middle = []
for i in group_scores.columns:
    CI = pd.DataFrame(conf_inter_mid[t], columns = ['Bootstrap'])
    if round(CI.Bootstrap.value_counts()[0]/1000*100) > 85:
        plt.plot(CU_SUM_Mid_pd[i], label = i, lw = 3, marker = markers[c], color = color[c])#+ ' with ' + str(round(CI.Bootstrap.value_counts()[0]/1000*100)) + '% CI'
        c = c + 1
        middle.append(i)
    t = t + 1
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
#plt.legend()
plt.title('CUSUM Chart for Higly Variable Middle Level ICCS Topics', fontsize = 14)
plt.grid(color='black', alpha=0.5, linestyle='dashed', linewidth=0.2)
plt.savefig('CUSUM Chart for Middle Level ICCS Topics.tiff', format='tiff', dpi=300, bbox_inches="tight")
plt.show()


# In[18]:

data = CU_SUM_Mid_pd[middle]
data.head()


# In[22]:

data.to_csv("D:\\Topic Modeling\\preprocessed corpus\\ICCS_Whole\\JoCS\\Final\\Results\\Highly variable middle level topics.csv")


# In[19]:

t = 0
for i in group_scores.columns:
    CI = pd.DataFrame(conf_inter_mid[t], columns = ['Bootstrap'])
    if round(CI.Bootstrap.value_counts()[0]/1000*100) > 85:
        m = list(abs(CU_SUM_Mid_pd[i]))
        s = np.sort(m)
        print (i, np.max(m), m.index(np.max(m)),  m.index(s[-2]), m.index(s[-3]), round(CI.Bootstrap.value_counts()[0]/1000*100))
    t = t + 1


# In[147]:

plt.figure(figsize=(12, 8))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Topic score', fontsize=14)
markers = ['+','o','*','p','x','s','d','^','v','>','<','p','h']
plt.xlim(2000.5, 2016.5)
for i in group_scores.columns:
    plt.plot(group_scores[i])
#plt.legend(fontsize=14)
plt.show()
len(group_scores.columns)


# In[7]:

N = len(group_scores.columns)
for i in range(len(group_scores.columns)):
    slp = [sp.stats.linregress(np.arange(N), np.array(group_scores.iloc[[i]][group_scores.columns]).flatten()).slope]
sns.distplot(slp)
for i in list(group_scores.columns):
    coefficients, residuals, _, _, _ = np.polyfit(range(len(group_scores[i])),group_scores,1,full=True)


# In[9]:

N


# In[144]:

variat = np.std(np.array(group_scores[group_scores.columns]), axis = 1) / np.average(np.array(group_scores[group_scores.columns]), axis = 1)
sns.distplot(variat)
sns.plt.title(np.mean(variat))


# In[145]:

filter = np.abs(variat) > 1
res_dy = group_scores.columns[filter]
res_dy


# In[100]:

plt.figure(figsize=(12, 8))
plt.xticks(fontsize=14, color = 'black')
plt.yticks(fontsize=14, color = 'black')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Normalized topic score', fontsize=14)
plt.xlim(2000.5, 2016.5)
#color_l = ["orange", "red", "green", "violet", "blue", "cyan", "yellow"]
c = 0
for i in res_dy:
    if i != 'HPC_Parallel':
        plt.plot(group_scores[i], marker = markers[c], lw = 3)
        c = c + 1
plt.legend(fontsize=14)
plt.title('Highly variable second level ICCS topics', fontsize=14, color = 'black')
plt.show()


# In[13]:

plt.figure(figsize=(12, 8))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Topic score', fontsize=14)
plt.xlim(2000.5, 2016.5)
c = 0
for i in group_scores.columns:
    if i not in res_dy:
        plt.plot(group_scores[i], '-o', lw = 3)
        c = c + 1
#plt.legend(fontsize=14)
plt.show()


# In[18]:

from collections import OrderedDict
S_res_Modeling = {}
for i in group_scores.columns:
    if i[:3] == 'Mod':
        S_res_Modeling.update({i:group_scores[i].sum()})
S_d_sorted_by_value_Modeling = OrderedDict(sorted(S_res_Modeling.items(), key=lambda x: x[1]))
plt.figure(figsize=(8, 6))
plt.xticks(fontsize=14, color = 'black')
plt.yticks(fontsize=14, color = 'black')
S_objects = list(S_d_sorted_by_value_Modeling.keys())
S_y_pos = np.arange(len(S_objects))    
S_res1 = list(S_d_sorted_by_value_Modeling.values())
plt.barh(S_y_pos, S_res1, color = 'green', lw = 2)
plt.yticks(S_y_pos, S_objects)
plt.title('ICCS second level topics\' shares', fontsize=14, color = 'black')
plt.show()


# In[31]:

from collections import OrderedDict
S_res_HPC = {}
for i in group_scores.columns:
    if i[:3] == 'Edu':
        S_res_HPC.update({i:group_scores[i].sum()})
S_d_sorted_by_value_HPC = OrderedDict(sorted(S_res_HPC.items(), key=lambda x: x[1]))
plt.figure(figsize=(8, 6))
plt.xticks(fontsize=14, color = 'black')
plt.yticks(fontsize=14, color = 'black')
S_objects = list(S_d_sorted_by_value_HPC.keys())
S_y_pos = np.arange(len(S_objects))    
S_res1 = list(S_d_sorted_by_value_HPC.values())
plt.barh(S_y_pos, S_res1, color = 'green', lw = 2, align = 'center')
plt.yticks(S_y_pos, S_objects)
plt.title('ICCS second level topics\' shares', fontsize=14, color = 'black')
plt.show()


# In[19]:

group_scores.columns


# In[155]:

plt.figure(figsize=(14,6))
plt.subplot(121)
plt.xticks(fontsize=20, color = 'black')
plt.yticks(fontsize=20, color = 'black')
#S_objects = list(S_d_sorted_by_value_Modeling.keys())
S_objects = [i[9:] for i in list(S_d_sorted_by_value_Modeling.keys())]
S_y_pos = np.arange(len(S_objects))    
S_res1 = list(S_d_sorted_by_value_Modeling.values())
plt.barh(S_y_pos, S_res1, color = 'green', lw = 2, align = 'center')
plt.yticks(S_y_pos, S_objects)
plt.title('ICCS Middle Level Topics, under Modeling', fontsize=20, color = 'black')
plt.grid(color='black', alpha=0.5, linestyle='dashed', linewidth=0.175)
plt.xlabel('Proportion of Middle Level Topics', fontsize=20)
plt.subplot(122)
plt.xticks(fontsize=20, color = 'black')
plt.yticks(fontsize=20, color = 'black')
#S_objects = list(S_d_sorted_by_value_HPC.keys())
S_objects = [i[4:] for i in list(S_d_sorted_by_value_HPC.keys())]
S_y_pos = np.arange(len(S_objects))    
S_res1 = list(S_d_sorted_by_value_HPC.values())
plt.barh(S_y_pos, S_res1, color = 'blue', lw = 2, align = 'center')
plt.yticks(S_y_pos, ["\n".join(
                              [" ".join(i.split("~")[:-1]),
                              "\n".join(i.split("~")[-1].split(" "))])
                   for i in S_objects])
plt.xlim(0.0,1.86)
plt.tight_layout()
plt.title('ICCS Middle Level Topics, under HPC', fontsize=20, color = 'black')
plt.grid(color='black', alpha=0.5, linestyle='dashed', linewidth=0.2)
plt.xlabel('Proportion of Middle Level Topics', fontsize=20)
plt.savefig('Middle Level ICCS Topics Share.tiff', format='tiff', dpi=300, bbox_inches="tight")
#plt.show()


# In[170]:

for i in list(S_d_sorted_by_value_HPC.keys()):
    print(i[4:])
[i[4:] for i in list(S_d_sorted_by_value_HPC.keys())]
len(list(S_d_sorted_by_value_HPC.keys()))


# In[32]:

few_share = ["Modeling_Data-driven", "Modeling_Finance", "Modeling_Transportation", "HPC_GPU", "Modeling_Data-driven","Modeling_ABM", "HPC_Fault tolerant"]
few_share2 = [ "eScience_Workflows","Modeling_Multi-scale", "Numerical_Solvers", "HPC_Parallel", "Machine Learning_Methods"]
c = 0
plt.figure(figsize=(8, 7))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Normalized Topic proportion', fontsize=13)
plt.xlim(2000.5, 2016.5)
for i in few_share:
    plt.plot(group_scores[i], marker = markers[c],lw = 2, markersize=10, color= color_l[c])
    c = c + 1
plt.legend(fontsize=14)
plt.title('Highly variable second level ICCS topics', fontsize=14, color = 'black')
plt.show()

