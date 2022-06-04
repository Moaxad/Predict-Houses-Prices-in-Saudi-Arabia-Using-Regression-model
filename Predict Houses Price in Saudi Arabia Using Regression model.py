#!/usr/bin/env python
# coding: utf-8

# In[120]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
#from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
#from sklearn.neural_network import MLPRegressor


# In[2]:


df = pd.read_csv("SA_Aqar.csv")


# # Let's clean the dataset

# In[3]:


#First, let's check of duplicate records in the dataset.
print(df.duplicated().sum())


# In[4]:


df.info()


# In[5]:


#Apperantly, 2197 records out of 3718 records are duplicates
#Removing duplicates records
df = df.drop_duplicates()


# In[6]:


df.info()


# # Feature engineering

# In[7]:


#All columns in this dataset are important except 'details' since it has uniuqe values in every record.
df.drop('details', inplace=True, axis=1)


# In[8]:


df.head()


# In[ ]:


#Before alculating the correlation coefficient, we need to deal with string values appearing in city, district and front.
#We will give all city, district and front values unique codes.


# In[9]:


ordEnc = OrdinalEncoder()
df["city_code"] = ordEnc.fit_transform(df[["city"]])
df["district_code"] = ordEnc.fit_transform(df[["district"]])
df["front_code"] = ordEnc.fit_transform(df[["front"]])


# In[10]:


#Reordering the columns in the dataframe.
df = df[['city','city_code','district','district_code','front','front_code','size'
         ,'property_age','bedrooms','bathrooms','livingrooms','kitchen','garage','driver_room',
           'maid_room','furnished','ac','roof','pool','frontyard',
         'basement','duplex','stairs','elevator','fireplace','price']]


# In[11]:


df.head(10)


# In[ ]:


#city codes:
# جدة = 3.0
# الرياض = 2.0
# الدمام = 1.0
# الخبر = 0.0

#front codes:
# ثلاث شوارع = 0.0 
# اربع شوارع = 1.0
# جنوب = 2.0
# جنوب شرقي = 3.0
# جنوب غربي = 4.0
# شرق = 5.0
# شمال = 6.0
# شمال شرقي = 7.0
# شمال غربي = 8.0
# غرب = 9.0

#districts codes from 0.0 - 173.0


# In[12]:


#Calculating the correlation coefficient
plt.figure(figsize=(18,16))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[ ]:


#We should take the features that has highest number comparing with (price) feature, which are: 
#From high values to lower: basement, fireplace, pool, livingrooms, elevator, driver_room, furnished.


# In[13]:


#Plot pairwise relationships
sns.pairplot(df)


# # We can see strong proportional relationships between bathrooms, property_age, bedrooms, kitchen, livingrooms, driver_room, maid_room garage and city_code (الرياض & جدة), front_yard, pool, basement, elevator, fire_place compared with price.

# In[20]:


#Taking price as a response variable (dependent variable)
y = df.price
y


# In[22]:


#Dropping string featrues and low correlation coefficient number features.
X = df.drop(['city','district','front','price', 'district_code','front_code','size','property_age'
             ,'bedrooms','bathrooms','kitchen','garage','frontyard','duplex','stairs','roof','city_code',
             'ac','maid_room'], axis=1)
X


# In[23]:


#80 20 split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[24]:


X_train.shape, y_train.shape


# In[25]:


X_test.shape, y_test.shape


# # Linear Regression model

# In[36]:


model_linear = linear_model.LinearRegression()
model_linear.fit(X_train, y_train)
y_pred = model_linear.predict(X_test)
x = r2_score(y_test, y_pred)
print(x)


# # We had 0.75 value for r^2 which does not mean the accuracy of the model since regression problems cannot be mesaured by accuracy.
# # So r^2 indicates the space between the line of x and y in the scatter polt below and the actual values in the test set.
# # 0.75 is not a very good value but for this dataset which it has a lot of problems and some of it is random values we can consider it as a satisfying result.

# # Scatter plot for Linear Regression

# In[37]:


sns.scatterplot(y_test, y_pred, alpha=0.5)

