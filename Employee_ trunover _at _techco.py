#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sys
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('C:/Users/Admin/Downloads/portfolio dataset/simulated_TECHCO_data.csv')


# ### Display Rows of Dataset

# In[3]:


df.head(10)


# In[4]:


df.tail(10)


# ### Dataset Overview

# In[5]:


print(df.size)
print(df.shape)
print(df.index)


# In[6]:


print(df.columns)


# In[7]:


print(sys.getsizeof(df))


# In[8]:


print(df.info())


# ### Statistical Summary about Dataframe

# In[9]:


df.describe()


# In[10]:


df[['training_score','logical_score','verbal_score','avg_literacy']].describe()


# ### Data Cleaning

# In[11]:


df.isnull()


# In[12]:


df.isnull().sum()


# In[13]:


df.duplicated()


# In[14]:


df.duplicated().sum()


# In[15]:


df.nunique()


# In[16]:


from timeit import default_timer as timer
start=timer()
df.info()
print(timer()-start)


# **Data Exploration**

# In[17]:


df['similar_language'].value_counts()


# In[18]:


df['similar_language'].unique()


# In[19]:


df['time'].value_counts()


# In[20]:


df['time'].unique()


# In[21]:


df['turnover'].value_counts()


# In[22]:


df['turnover'].unique()


# In[23]:


df['is_male'].unique()


# In[24]:


df['is_male'].value_counts()


# In[25]:


df.dtypes


# **Analyzing Key Rows and Columns**

# In[26]:


df.iloc[5]


# In[27]:


df.iloc[8]


# In[28]:


df.iloc[1:10]


# In[29]:


df.iloc[: ,1:8]


# **Top 5 Highest Training Scores**

# In[30]:


df['training_score'].nlargest(5)


# **Location-Based Average Scores Analysis for Training and Skills**

# In[31]:


location_averages = df.groupby('location_age')[['training_score', 'avg_literacy', 'verbal_score', 'logical_score']].mean()
sorted_location_averages = location_averages.sort_index()
print(sorted_location_averages)


# **Comparing Top and Bottom Scores by Gender**

# In[32]:


print(df.groupby('is_male')['training_score','avg_literacy','verbal_score','logical_score'].max())
print(df.groupby('is_male')['training_score','avg_literacy','verbal_score','logical_score'].min())


# **Counts of Male and Female Employees by Literacy and Training Score**

# In[33]:


df[['is_male','avg_literacy','training_score']].value_counts()


# **Best Performing Employees by Gender: Max Scores Overview**

# In[34]:


grouped = df.groupby('is_male')
result = grouped.apply(lambda x: x.loc[x[['training_score', 'avg_literacy', 'logical_score', 'verbal_score']].idxmax()])
result = result.reset_index(drop=True)
print(result[['is_male', 'emp_id', 'training_score', 'avg_literacy', 'logical_score', 'verbal_score']])


# **Data Analysis: Employees Who Stayed and Left**

# In[35]:


df[df['turnover']=='Stayed']


# In[36]:


df[df['turnover']=='Left']


# **List of Employee IDs with Turnover 'Left**

# In[37]:


df[df['turnover']=='Left']['emp_id']


# **Unique Employee IDs with Similar Language Score of 100**

# In[38]:


print(df[df['similar_language']==100]['emp_id'].unique())
print('No.of Emp_id who have similar_language=100 : ',df[df['similar_language']==100]['emp_id'].nunique())


# **Selected Employee Data: Key Metrics**

# In[39]:


df[['emp_id','training_score','location_age','similar_language']]


# **Distinct Data for Stayed Turnover: Employee Details and Metrics**

# In[40]:


df[df['turnover']=='Stayed'][['emp_id','training_score','location_age','distance','similar_language','turnover']].drop_duplicates()


# **Distinct Data for Left Turnover: Employee Details and Metrics**

# In[41]:


df[df['turnover']=='Left'][['emp_id','time','is_male','training_score','location_age','distance','similar_language','turnover']].drop_duplicates()


# **Analysis of Stayed Turnover with Time Greater Than or Equal to 20**

# In[42]:


df[(df['time']>=20) & (df['turnover']=='Stayed')]


# **Extracting Time Data for Employees Who Stayed**

# In[43]:


df[df['turnover']=='Stayed']['time']


# **Pivot Table Analysis for Male Employees: Literacy and Training Scores**

# In[44]:


pivot_table = df[df['is_male'] == 1].pivot_table(index='is_male',values=['avg_literacy', 'training_score'],aggfunc=['max', 'min', 'mean'])
print(pivot_table)


# **Pivot Table Analysis for Female Employees: Literacy and Training Scores**

# In[45]:


pivot_table = df[df['is_male'] == 0].pivot_table(index='is_male',values=['avg_literacy', 'training_score'],aggfunc=['max', 'min', 'mean'])
print(pivot_table)


# ### Visualization

# **Distribution oF Gender**

# In[46]:


plt.figure(figsize=(5, 5))
sns.countplot(x='is_male', data=df)
plt.title('Gender Distribution')
plt.xlabel('Gender (0=Female, 1=Male)')
plt.ylabel('Count')
plt.show()


# **Trunover Rate by Gender**

# In[47]:


plt.figure(figsize=(5, 6))
sns.countplot(x='turnover', hue='is_male', data=df)
plt.title('Turnover by Gender')
plt.xlabel('Turnover')
plt.ylabel('Count')
plt.legend(title='Gender', loc='upper right', labels=['Female', 'Male'])
plt.show()


# **Turnover Rate by Location Age**

# In[48]:


plt.figure(figsize=(8, 5))
sns.lineplot(x='location_age', y='turnover', data=df, ci=None)
plt.title('Turnover Rate by Location Age')
plt.xlabel('Location Age')
plt.ylabel('Turnover Rate')
plt.show()


# **Distribution of Scores (Training, Logical, Verbal)**

# In[49]:



plt.figure(figsize=(14, 6))
plt.subplot(1, 3, 1)
sns.histplot(df['training_score'], kde=True, color='blue')
plt.title('Distribution of Training Course Score')

plt.subplot(1, 3, 2)
sns.histplot(df['logical_score'], kde=True, color='green')
plt.title('Distribution of Logical Score')

plt.subplot(1, 3, 3)
sns.histplot(df['verbal_score'], kde=True, color='red')
plt.title('Distribution of Verbal Score')

plt.tight_layout()
plt.show()


# **Average Literacy vs. Turnover**

# In[50]:


plt.figure(figsize=(7, 5))
sns.boxplot(x='turnover', y='avg_literacy', data=df)
plt.title('Average Literacy vs Turnover')
plt.xlabel('Turnover')
plt.ylabel('Average Literacy')
plt.show()


# **Turnover by Time (Months at Company)**

# In[51]:


plt.figure(figsize=(8, 5))
sns.lineplot(x='time', y='turnover', data=df, ci=None)
plt.title('Turnover Rate by Time (Months at Company)')
plt.xlabel('Months at Company')
plt.ylabel('Turnover Rate')
plt.show()


# **Distance from Production Center vs. Turnover**

# In[52]:



plt.figure(figsize=(8, 4))
sns.scatterplot(x='distance', y='turnover', data=df)
plt.title('Distance vs Turnover')
plt.xlabel('Distance from Production Center')
plt.ylabel('Turnover')
plt.show()


# **Language Similarity and Turnover**

# In[53]:


plt.figure(figsize=(7, 5))
sns.barplot(x='similar_language', y='turnover', data=df)
plt.title('Language Similarity vs Turnover')
plt.xlabel('Language Similarity')
plt.ylabel('Turnover Rate')
plt.show()


# **Overall Turnover Rate**

# In[54]:


plt.figure(figsize=(5, 5))
turnover_counts = df['turnover'].value_counts()
plt.pie(turnover_counts, labels=['Stayed', 'Left'], autopct='%1.1f%%', startangle=140, colors=['skyblue', 'salmon'])
plt.title('Overall Turnover Rate')
plt.show()


# **Correlation Matrix**

# In[55]:


plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

