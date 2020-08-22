#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


df = pd.read_csv("ipl_matches.csv")
df.head()


# In[3]:


## -----data cleaning------
## remove unwanted columns

columns_to_remove = ['mid','batsman','bowler','striker','non-striker']
df.drop(labels=columns_to_remove,axis=1,inplace=True)


# In[4]:


df.head()


# In[5]:


df['bat_team'].unique()


# In[6]:


### keeping only consistant team

consistant_team = ['Kolkata Knight Riders','Chennai Super Kings','Rajasthan Royals', 'Mumbai Indians',
                  'Kings XI Punjab', 'Royal Challengers Bangalore','Delhi Daredevils','Sunrisers Hyderabad',]


# In[7]:


df = df[(df['bat_team'].isin(consistant_team)) & (df['bowl_team'].isin(consistant_team))]


# In[8]:


df.head()


# In[9]:


df = df[df['overs']>=5.0]


# In[10]:


df.head()


# In[11]:


### converting the 'date' column from string to datetime object

from datetime import datetime
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%d-%m-%Y'))


# In[12]:


df.head()


# In[13]:


print(df['bat_team'].unique())
print(df['bowl_team'].unique())


# In[14]:


###-------data processing-------
### converting the categoral features using one hot encoding

encoded_df = pd.get_dummies(data=df,columns=['venue','bat_team','bowl_team'])
encoded_df.head()


# In[15]:


encoded_df.columns


# In[16]:


### rearranging the columns

encoded_df = encoded_df[['date','runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5',
       'venue_Barabati Stadium', 'venue_Brabourne Stadium',
       'venue_Buffalo Park', 'venue_De Beers Diamond Oval',
       'venue_Dr DY Patil Sports Academy',
       'venue_Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
       'venue_Dubai International Cricket Stadium', 'venue_Eden Gardens',
       'venue_Feroz Shah Kotla',
       'venue_Himachal Pradesh Cricket Association Stadium',
       'venue_Holkar Cricket Stadium',
       'venue_JSCA International Stadium Complex', 'venue_Kingsmead',
       'venue_M Chinnaswamy Stadium', 'venue_MA Chidambaram Stadium, Chepauk',
       'venue_Maharashtra Cricket Association Stadium',
       'venue_New Wanderers Stadium', 'venue_Newlands',
       'venue_OUTsurance Oval',
       'venue_Punjab Cricket Association IS Bindra Stadium, Mohali',
       'venue_Punjab Cricket Association Stadium, Mohali',
       'venue_Rajiv Gandhi International Stadium, Uppal',
       'venue_Sardar Patel Stadium, Motera', 'venue_Sawai Mansingh Stadium',
       'venue_Shaheed Veer Narayan Singh International Stadium',
       'venue_Sharjah Cricket Stadium', 'venue_Sheikh Zayed Stadium',
       "venue_St George's Park", 'venue_Subrata Roy Sahara Stadium',
       'venue_SuperSport Park', 'venue_Wankhede Stadium',
       'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils',
       'bat_team_Kings XI Punjab', 'bat_team_Kolkata Knight Riders',
       'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
       'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
       'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils',
       'bowl_team_Kings XI Punjab', 'bowl_team_Kolkata Knight Riders',
       'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
       'bowl_team_Royal Challengers Bangalore',
       'bowl_team_Sunrisers Hyderabad', 'total']]


# In[17]:


encoded_df.head()


# In[18]:


### Splitting the data into train and test dataset

x_train = encoded_df.drop(labels=['total'],axis=1)[encoded_df['date'].dt.year <=2016]
x_test = encoded_df.drop(labels=['total'],axis=1)[encoded_df['date'].dt.year >=2017]


# In[19]:


y_train = encoded_df[encoded_df['date'].dt.year <=2016]['total'].values
y_test = encoded_df[encoded_df['date'].dt.year >=2017]['total'].values


# In[20]:


### removing the 'date' column

x_train.drop(labels='date',axis=1,inplace=True)
x_test.drop(labels='date',axis=1,inplace=True)


# In[25]:


### -----Model Building-----
### Linear Regression

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)


# In[26]:


### creating a pickel file for the classifier

import pickle
filename = 'model.pkl'
pickle.dump(regressor,  open(filename, 'wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




