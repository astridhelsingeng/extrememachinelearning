#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


missing_values = ["n/a", "na", "--", "-", " ", "NAN", "nan", "Nan", "NaN"] # Additional NaN possibilities
df = pd.read_csv("archive/aug_train.csv", na_values = missing_values)

df.head()


# In[3]:


df['major_discipline'].unique()


# In[4]:


df = df.drop("enrollee_id", axis=1)


# In[5]:


df = df.dropna(subset=['gender', 'enrolled_university', 'education_level', 'experience', 'last_new_job'], axis = 0, how='any')
df.reset_index(drop=True, inplace=True)


# In[6]:


df.isnull().sum()


# In[7]:


# Create binary values for relevant experience
for i in df["relevent_experience"]:
    if(i == "Has relevent experience"):
        df["relevent_experience"][df["relevent_experience"]==i]=1
    if(i == "No relevent experience"):
        df["relevent_experience"][df["relevent_experience"]==i]=0
        
df["relevent_experience"] = df["relevent_experience"].astype('int')


# In[8]:


# Change experience above 20 and less than 1
for i in df["experience"]:
    if(i==">20"):
        df["experience"][df["experience"]==i]=21
    if(i == "<1"):
        df["experience"][df["experience"]==i]=0

df["experience"] = df['experience'].astype('int')


# In[13]:


for i in range(len(df)):
    if (df['education_level'][i] == 'High School') or (df['education_level'][i] == "Primary School"):
        df['major_discipline'][i] = 'Not applicable'
df = df.dropna(subset=['major_discipline'], axis = 0, how='any')

df["company_type"] = df["company_type"].fillna("Undefined")


# In[14]:


df['major_discipline'].unique()


# In[15]:


for i in df["company_size"]:
    if(i=="10/49"):
        df["company_size"][df["company_size"]==i]="10-49"
    if(i == "<10"):
        df["company_size"][df["company_size"]==i]="1-9"
    if(i == "10000+"):
        df["company_size"][df["company_size"]==i]="10000-100000"

df["company_size"] = df["company_size"].fillna("0")

df.head()


for i in df["last_new_job"]:
    if(i==">4"):
        df["last_new_job"][df["last_new_job"]==i]=5
    if(i == "never"):
        df["last_new_job"][df["last_new_job"]==i]=0

df["last_new_job"] = df["last_new_job"].fillna(0)       

df["last_new_job"] = df['last_new_job'].astype('int')
df["last_new_job"].unique()

df.reset_index(drop=True, inplace=True)

for i in range(len(df["city"])):
    df["city"][i] = df["city"][i].replace("city_","")

df["city"] = df['city'].astype('int')


for column in df.columns:
    if df[column].dtype == "object":
        df[column] = df[column].astype("category")
        df[column] = df[column].cat.codes.astype("int64")


df.info()
df.head()


##


import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("seaborn-colorblind")
sns.pairplot(df, hue="target")
plt.show()



##

