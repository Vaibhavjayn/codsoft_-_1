#!/usr/bin/env python
# coding: utf-8

# # TASK: TITANIC SURVIVAL PREDICTION

# DOMAIN : DATA SCIENCE

# ## Importing the Libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ## Data Collection and Processing

# In[2]:


# Creating a DataFrame using CSV file
data=pd.read_csv("Titanic-Dataset.csv")


# In[3]:


# print first five rows of the DataFrame
data.head()


# In[4]:


# print the number of rows and columns
data.shape


# In[34]:


# print columns names of the DataFrame
data.columns


# In[6]:


# checking the null values
data.isnull().sum()


# ## Handling Null Values

# In[7]:


#Drop the column Cabin from dataFrame
data=data.drop(columns=["Cabin","Name"],axis=1)


# In[8]:


# Filling the missing values in 'Age' column by taking Mean.
data["Age"].fillna(data["Age"].mean(),inplace=True)


# In[9]:


data=data.dropna()


# In[10]:


data.isnull().sum()


# ##  Statistical analysis

# In[11]:


data.describe()


# In[12]:


# Categorise the count of Survived and Not Survived people
data["Survived"].value_counts()


# In[13]:


# Categorise the count of male and female
data["Sex"].value_counts()


# In[14]:


data["Pclass"].value_counts()


# ## Data Visualization

# In[15]:


a=data["Survived"].value_counts()
x=["Not Survived","Survived"]
y=[a[i] for i in range(0,2)]
plt.bar(x,y)
plt.title("No. of people survived and not survived")
plt.show()


# In[16]:


b=data["Sex"].value_counts()
x=["Male","Female"]
y=[b[i] for i in range(0,2)]
plt.bar(x,y)
plt.title("No. of males and females")
plt.show()


# In[17]:


x=["Male","Female"]
c=data[data["Survived"]==0]["Sex"].value_counts()
d=data[data["Survived"]==1]["Sex"].value_counts()
y_male=[c[0],c[1]]
y_female=[d[1],d[0]]
x_axis = np.arange(len(x)) 
plt.bar(x_axis - 0.2,y_male, 0.4, label = 'Not Survived') 
plt.bar(x_axis + 0.2,y_female, 0.4, label = 'Survived') 
plt.xticks(x_axis, x)
plt.legend() 
plt.show() 


# In[18]:


# creating a Countplot for the Pclass Column
sns.countplot(data =data,x ='Pclass')


# In[19]:


# Plotting number of survivors based on Pclass
sns.countplot(x = 'Pclass', hue = 'Survived', data =data)


# ## Performing Encoding over Columns

# In[20]:


data['Embarked'].value_counts()


# In[21]:


# Coverting columns into Categorical Columns
data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0, 'C':1, 'Q':2}}, inplace = True)


# In[22]:


data.head()


# ## Splitting the new data into Features and Target

# In[23]:


X =data.drop(columns = ['PassengerId', 'Ticket','Survived'],axis = 1)
Y=data['Survived']


# In[24]:


X.head()


# In[25]:


Y.head()


# ## Splitting the data into Testing and Traning data

# In[26]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.25,random_state = 3)


# In[27]:


print(X.shape, X_train.shape, X_test.shape)


# ## Model Training

# In[28]:


model=LogisticRegression(max_iter=200)


# In[29]:


model.fit(X_train,Y_train)


# ## Model Evaluation

# In[30]:


training_prediction = model.predict(X_train)
training_prediction


# In[31]:


# Accuracy of training data
training_accuracy = accuracy_score(Y_train,training_prediction)
print('Accuracy Score of Training data:',training_accuracy)


# In[32]:


testing_prediction = model.predict(X_test)
testing_prediction


# In[33]:


# Accuracy of testing data
testing_accuracy = accuracy_score(Y_test,testing_prediction)
print('Accuracy Score of Testing data:',testing_accuracy)

