#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv('creditcard.csv')


# In[3]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8), dpi=85)
x = [0, 1]
bar_list = plt.bar(x, [len(df[df['Class']==0]), len(df[df['Class'] == 1])])
bar_list[1].set_color('orange')
plt.xticks(x, ('0', '1'))
plt.show()


def ration(x_1, x_0):
    x_1_num = len(x_1)
    x_0_num = len(x_0)
    x_1_percent = (x_1_num*100)/(x_1_num+x_0_num)
    x_0_percent = 100 - x_1_percent
    return x_0_percent, x_1_percent
fraud_percent, norm_percent = ration( df[df['Class']==0], df[df['Class'] == 1] )
print('Norm transaction {} %\n'.format(norm_percent))
print('Fraud transaction {} %\n'.format(fraud_percent))


# In[4]:


#underfitting 
import random
df[df['Class'] == 1]
X = df.iloc[:,:-1]
y = df['Class']
def sample_together(n, X, y):
    rows = random.sample(np.arange(0,len(X.index)).tolist(),n)
    return X.iloc[rows,], y.iloc[rows,]

def undersample(X, y):
    y_min = y[y == 1]
    y_max = y[y == 0]
    X_min = X.filter(y_min.index,axis = 0)
    X_max = X.filter(y_max.index,axis = 0)

    X_under, y_under = sample_together(len(y_min.index), X_max, y_max)
    
    X = pd.concat([X_under, X_min])
    y = pd.concat([y_under, y_min])
    return X, y

X_new, y_new = undersample(X, y)


plt.figure(figsize=(10, 8), dpi=50)
x = [0, 1]
bar_list = plt.bar(x, [len(y_new[y_new ==0]), len(y_new[y_new == 1])])
bar_list[1].set_color('orange')
plt.xticks(x, ('0', '1'))
plt.show()


# In[103]:





# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=1)
X_train.describe()
X_test.describe()


# In[ ]:





# In[6]:


names_x_train = X_train.columns

#normalisation train data
x_train_mean = X_train.describe().loc['mean']
x_train_std = X_train.describe().loc['max'] - X_train.describe().loc['min']

#normalisation test data
x_test_mean  =  X_test.describe().loc['mean']
x_test_std = X_test.describe().loc['max'] - X_test.describe().loc['min']

for col in names_x_train:
    X_train[col] = (X_train[col] - x_train_mean[col])/x_train_std[col]
    
    X_test[col] = (X_test[col] - x_test_mean[col])/x_test_std[col]
X_train.describe()


# In[19]:


data_train = pd.concat([X_train, y_train], axis=1, sort=False)
names = data_train.columns

data_train_fraud = (data_train[data_train['Class'] == 1])
data_train_norm = (data_train[data_train['Class'] == 0])

data_train = pd.concat([data_train_fraud[names[:-1]],data_train_norm[names[:-1]]])

data_test_y = (pd.concat([X_test, y_test],axis=1, sort = False)).iloc[:150]




# In[23]:


g = 0.1
data_test = data_test_y[names[:-1]]
def pnn(g,  data_train, data_test):
    N = len(data_train)
    m = len(data_test) 
    answers = [0]*m
    for i in range(m):
        f_sum_list = [0]*N
        for j in range(N):
            f_sum_list[j] = (np.e**(-1*sum((data_train.iloc[j] - data_test.iloc[i])**2)/(g**2)))

        exam_num = [len(data_train_norm), len(data_train_fraud)]
        q_n = np.mean(f_sum_list[:exam_num[0]])
        q_f = np.mean(f_sum_list[exam_num[0]:exam_num[1]+exam_num[0]])

        q_list = [q_n, q_f]
        index = q_list.index(max(q_list))

        if index == 0:
            answer = 1
        elif index == 1:
            answer = 0
        answers[i] = answer
    
    data_test_res = data_test_y[:m]

    data_test_res.loc[:,'pnn_class'] = answers
    len_res_pnn = len(data_test_res[data_test_res['Class'] == data_test_res['pnn_class']])

    #find_error
    accure_perсent = (len_res_pnn*100)/m
    
    return data_test_res, accure_perсent

data_res, accure_perсent = pnn(g,data_train, data_test)
print("Accuracy percent = {}".format(accure_perсent))
print("Total test num = {}".format(len(data_test)))
print("Predicted num  = {}".format(len(data_res[data_res['Class'] == data_res['pnn_class']])))


# In[22]:


data_res.iloc[10:20]


# In[ ]:




