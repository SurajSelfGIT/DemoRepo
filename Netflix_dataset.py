#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Netflix prize dataset
#importing the necessary libraries for importing the dataset in jupyter notebook
#100M ratings 17770 movies 480,000 users 4 datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# In[5]:


# Reading dataset file
netflix_dataset = pd.read_csv('D:\Python Suraj\Subject wise\Project - Netflix\combined_data_1.txt\combined_data_1.txt',
                              header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])
netflix_dataset.head()


# In[ ]:


netflix_dataset.head()


# In[6]:


netflix_dataset.dtypes


# In[7]:


netflix_dataset['Rating']=netflix_dataset['Rating'].astype(float)


# In[8]:


netflix_dataset.dtypes


# In[9]:


netflix_dataset.shape


# In[10]:


#To find out how many people have rated the movies as 1, 2, 3,4,5 stars ratings to the movies
stars=netflix_dataset.groupby('Rating')['Rating'].agg(['count'])


# In[11]:


stars


# In[12]:


#to claculate how many movies we are having in the dataset
movie_count=netflix_dataset.isnull().sum()
movie_count


# In[13]:


movie_count=netflix_dataset.isnull().sum()[1]
movie_count


# In[14]:


#get the customer count with NaN values
customer_count=netflix_dataset['Cust_Id'].nunique()


# In[15]:


customer_count


# In[16]:


#without NaN values
customer_count=netflix_dataset['Cust_Id'].nunique()-movie_count
customer_count


# In[17]:


#get the total number of ratings given by the customers
rating_count=netflix_dataset['Cust_Id'].count()-movie_count
rating_count


# In[ ]:


ax=stars.plot(kind='barh', legend=False, figsize=(15,10))
plt.title(f'Total pool: {movie_count} Movies, {customer_count} Customers, {rating_count} ratings given', fontsize=20)
plt.grid(True)


# In[ ]:


#add another column that will have movie id
#first of all we will be calculating how many null values I am having in the ratings column
df_nan=pd.DataFrame(pd.isnull(netflix_dataset.Rating))


# In[ ]:


df_nan.head()


# In[ ]:


df_nan=df_nan[df_nan['Rating']==True]


# In[ ]:


df_nan.shape


# In[ ]:


df_nan.head()


# In[ ]:


#now we will reset the index as the column
df_nan=df_nan.reset_index()


# In[ ]:


df_nan.head()


# In[ ]:


#now we will create a numpy array that will contain 1 from values 0 to 547, 2 from 548 to 693 and so on
movie_np=[]
movie_id=1
for i, j in zip(df_nan['index'][1:], df_nan['index'][:-1]):
    temp=np.full((1, i-j-1), movie_id)
    movie_np=np.append(movie_np, temp)
    movie_id+=1
    
#account for last record and corresponding length
#numpy approach
last_record=np.full((1, len(netflix_dataset)-df_nan.iloc[-1,0]-1), movie_id)#movie id will be 4499
movie_np=np.append(movie_np, last_record)
print(f'Movie numpy: {movie_np}')
print(f'Length: {len(movie_np)}')


# In[ ]:


#working
x=zip(df_nan['index'][1:], df_nan['index'][:-1])


# In[ ]:


tuple(x)


# In[ ]:


temp=np.full((1,547), 1)


# In[ ]:


print(temp)


# In[ ]:


netflix_dataset=netflix_dataset[pd.notnull(netflix_dataset['Rating'])]
netflix_dataset['Movie_Id']=movie_np.astype(int)
netflix_dataset['Cust_Id']=netflix_dataset['Cust_Id'].astype(int)
print("Now the dataset will look like: ")
netflix_dataset.head()


# In[ ]:


#now we will remove all the users that have rated less movies and 
#also all those movies that has been rated less in numbers
f=['count','mean']


# In[ ]:


dataset_movie_summary=netflix_dataset.groupby('Movie_Id').agg(f)


# In[ ]:


dataset_movie_summary


# In[ ]:


dataset_movie_summary=netflix_dataset.groupby('Movie_Id')['Rating'].agg(f)


# In[ ]:


dataset_movie_summary


# In[ ]:


#now we will store all the movie_id indexes in a variable dataset_movie_summary.index and convert the datatype to int
dataset_movie_summary.index=dataset_movie_summary.index.map(int)


# In[ ]:


#now we will create a benchmark 
movie_benchmark=round(dataset_movie_summary['count'].quantile(0.7),0)
movie_benchmark


# In[ ]:


dataset_movie_summary['count']


# In[ ]:


drop_movie_list=dataset_movie_summary[dataset_movie_summary['count']<movie_benchmark].index
drop_movie_list


# In[ ]:


#now we will remove all the users that are in-active
dataset_cust_summary=netflix_dataset.groupby('Cust_Id')['Rating'].agg(f)
dataset_cust_summary


# In[ ]:


dataset_cust_summary.index=dataset_cust_summary.index.map(int)


# In[ ]:


cust_benchmark=round(dataset_cust_summary['count'].quantile(0.7),0)
cust_benchmark


# In[ ]:


drop_cust_list=dataset_cust_summary[dataset_cust_summary['count']<cust_benchmark].index
drop_cust_list


# In[ ]:


#we will remove all the customers and movies that are below the benchmark 
print('The original dataframe has: ', netflix_dataset.shape, 'shape')


# In[ ]:


netflix_dataset=netflix_dataset[~netflix_dataset['Movie_Id'].isin(drop_movie_list)]
netflix_dataset=netflix_dataset[~netflix_dataset['Cust_Id'].isin(drop_cust_list)]
print('After the triming, the shape is: {}'.format(netflix_dataset.shape))


# In[ ]:


netflix_dataset.head()


# In[ ]:


#now we will prepare the dataset for SVD and it takes the matrix as the input
# so for input, we will convert the dataset into sparse matrix
#4499 movies
df_p = pd.pivot_table(netflix_dataset, values='Rating', index='Cust_Id', columns='Movie_Id')
print(df_p.shape)


# In[ ]:


df_p.head()


# In[ ]:


df_title=pd.read_csv('/content/drive/MyDrive/Untitled folder/SVD/movie_titles.csv', encoding='ISO-8859-1', header=None, names=['Movie_Id','Year','Name' ])
df_title.set_index('Movie_Id', inplace=True)


# In[ ]:


df_title.head(10)


# In[ ]:


#model building
get_ipython().system('pip install scikit-surprise')
import math
import re
from scipy.sparse import csr_matrix
import seaborn as sns
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate


# In[ ]:


#help us to read the dataset for svd algo
reader=Reader()


# In[ ]:


#we only work with top 100K rows for quick runtime
data=Dataset.load_from_df(netflix_dataset[['Cust_Id','Movie_Id','Rating']][:100000], reader)


# In[ ]:


svd=SVD()
cross_validate(svd, data, measures=['RMSE','MAE'], cv=3, verbose=True)
#for 1st fold- 1,2,3,4,5


# In[ ]:


netflix_dataset.head()


# In[ ]:


#so first we take user 712664 and we try to recommend some movies based on the past data
dataset_712664=netflix_dataset[(netflix_dataset['Cust_Id'] ==712664)& (netflix_dataset['Rating']==5)]
dataset_712664=dataset_712664.set_index('Movie_Id')
dataset_712664=dataset_712664.join(df_title)['Name']
dataset_712664


# In[ ]:


#now we will build the recommendation algorithm
#first we will make a shallow copy of the movie_titles.csv file so that we can change the values in the copied dataset, not in the actual dataset

user_712664=df_title.copy()
user_712664


# In[ ]:


user_712664=user_712664.reset_index()
user_712664


# In[ ]:


user_712664=user_712664[~user_712664['Movie_Id'].isin(drop_movie_list)]
user_712664


# In[ ]:


#now we will train our algorithm with the whole dataset
data=Dataset.load_from_df(netflix_dataset[['Cust_Id','Movie_Id','Rating']], reader)


# In[ ]:


#building the trainset using surprise package
trainset=data.build_full_trainset()
svd.fit(trainset)


# In[ ]:


user_712664['Estimate_Score']=user_712664['Movie_Id'].apply(lambda x: svd.predict(712664, x).est)
user_712664=user_712664.drop('Movie_Id', axis=1)


# In[ ]:


user_712664=user_712664.sort_values('Estimate_Score')
print(user_712664.head(10))


# In[ ]:


user_712664=user_712664.sort_values('Estimate_Score', ascending=False)
print(user_712664.head(10))


# In[ ]:




