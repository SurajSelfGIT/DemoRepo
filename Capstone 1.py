#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import itertools
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data=pd.read_csv(r"D:\Python Suraj\Subject wise\Projects\Capstone project\Walmart.csv")


# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


data.head(20)


# In[6]:


data.corr()


# In[7]:


data.Store.value_counts().to_frame().sort_index()


# In[7]:


data['Date']=pd.to_datetime(data['Date'], errors='ignore', dayfirst=True)


# In[8]:


data.head(20)


# In[9]:


data.info()


# In[10]:


round(data.corr(),3)


# In[11]:


def get_store_df(store_id):
    store=pd.DataFrame()
    store["sales"]=data[data["Store"]==store_id].Weekly_Sales
    store.index=data[data["Store"]==store_id].Date
    return store


# In[12]:


store1=get_store_df(1)


# In[13]:


store1


# In[14]:


store1.info()


# In[15]:


_, ax=plt.subplots(figsize=(25,8))
sns.boxplot(x=store1.index.month[0:100], y=store1.sales[0:100], ax=ax)
plt.title("Monthly sales for STORE-1")
plt.xlabel("Month")
plt.ylabel("Amount")
plt.xticks(rotation=50)
plt.grid()


# In[16]:


_, ax=plt.subplots(figsize=(25,8))
sns.barplot(x=store1.index[0:100], y=store1.sales[0:100], ax=ax)
plt.title("Weekly sales for 143 weeks for STORE-1")
plt.xlabel("Weekly dates")
plt.ylabel("Sales Amount")
plt.xticks(rotation=50)
plt.grid()


# In[17]:


store9=get_store_df(9)
store18=get_store_df(18)
store27=get_store_df(27)
store36=get_store_df(36)
store45=get_store_df(45)


# In[18]:


store1["sales"].plot(figsize=(15, 6), legend=True, color = 'turquoise')
store9["sales"].plot(figsize=(15, 6), legend=True, color = 'salmon')
store18["sales"].plot(figsize=(15, 6), legend=True, color = 'red')
store27["sales"].plot(figsize=(15, 6), legend=True, color = 'green')
store36["sales"].plot(figsize=(15, 6), legend=True, color = 'blue')
store45["sales"].plot(figsize=(15, 6), legend=True, color = 'maroon')
plt.ylabel('Weekly Sales')
plt.legend(['Store-1', 'Store-9', 'Store-18', 'Store-27', 'Store-36', 'Store-45'], loc="upper left")
plt.title('DIFFERENT STORES SALES COMPARISION', fontsize = '16')
plt.grid()
plt.show()


# In[19]:


from statsmodels.tsa.stattools import adfuller
result = adfuller(store1['sales'])
if result[1]>0.05:
    print('not stationary')
else:
    print("stationary")


# In[20]:


round(result[1],3)


# In[21]:


y1=store1['sales']

p = d = q = range(0, 5)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


# In[22]:


import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Best SARIMAX order - Lowest AIC value  (IT WILL TAKE LONG TIME)

# for param in pdq:
#    for param_seasonal in seasonal_pdq:
#        try:
#            mod = sm.tsa.statespace.SARIMAX(y1,
#                                            order=param,seasonal_order=param_seasonal,
#                                            enforce_stationarity=False,
#                                            enforce_invertibility=False)`
#            results = mod.fit()
#            print('ARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,results.aic))
#        except: 
#            continue
# In[23]:


mod = sm.tsa.statespace.SARIMAX(y1,
                                order=(4, 4, 3),
                                seasonal_order=(1, 1, 0, 52),   #enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])


# In[24]:


pred_uc = results.get_forecast(steps=12)

# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()


# In[25]:


round(pred_uc.predicted_mean,2)


# In[26]:


def forecast_12_week(y):
    model = sm.tsa.statespace.SARIMAX(y,
                                order=(4, 4, 3),
                                seasonal_order=(1, 1, 0, 52),   #enforce_stationarity=False,
                                enforce_invertibility=False)

    results = model.fit()
    predictions = results.get_forecast(steps=12)
    return predictions.predicted_mean.to_list()


# In[27]:


Store_pred_dict = defaultdict(list)
for k in data['Store'].unique():
    sales_store = get_store_df(k)
    y = sales_store.sales
    predicted_list = forecast_12_week(y)
    Store_pred_dict['Store-'+str(k)] = predicted_list


# In[28]:


Store_pred_dict_df = pd.DataFrame(Store_pred_dict)


# In[33]:


Store_pred_dict_df.loc[: , "Store-31":"Store-40"]


# In[ ]:




