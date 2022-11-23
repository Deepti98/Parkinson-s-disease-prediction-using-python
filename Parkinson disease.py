#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix


# # Reading the dataset

# In[2]:


df=pd.read_csv("parkinsons.data")


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.isnull().sum()#There is no null values in the dataframe


# In[6]:


#We can find the presence of missing values at the same time view the type of columns by using df.info()
df.info()


# In[7]:


df.describe()#Describe method is used for calculating the some statistical data like percentile,mean and std deviation of the 
#numerical values of series or dataframe


# In[8]:


df.columns#Display the columns name of dataframe


# In[9]:


df['status']#Status columns is a target column


# In[10]:


#Status for parkinson's=zero(0) and for healthy =1


# # Visulization

# In[11]:


plt.figure(figsize=(10,6))
df.status.hist()
plt.xlabel("status")
plt.ylabel("Frequencies")
plt.plot()
#The dataset has highest number of patients affected by parkinsons disease


# In[12]:


plt.figure(figsize=(10,6))
sns.barplot(x="status",y="NHR",data=df)
#The patients effected with Parkinsons disease have high NHR that is the measure of ratio noise to total components of voice


# In[13]:


plt.figure(figsize=(10,6))
sns.barplot(x="status",y="HNR",data=df)
#The patients effected with Parkinsons disease have high HNR that is the measure of ratio noise to total components of voice


# In[14]:


plt.figure(figsize=(10,6))
sns.barplot(x="status",y="RPDE",data=df)
#The nonlinear dynamical complexity measure RPDE is high in the patients effected with Parkinson's disease


# # Distribution plot

# In[15]:


rows=3
cols=7
fig,ax=plt.subplots(nrows=rows,ncols=cols,figsize=(16,4))
col=df.columns
index=1
for i in range(rows):
    for j in range(cols):
        sns.distplot(df[col[index]],ax=ax[i][j])
        index=index+i
        
plt.tight_layout()


# ## A distribution plot displays a distribution and a range of a set of numeric values plotted against a dimensions

# In[16]:


df.drop(['name'],axis=1,inplace=True)#Removing name columns for machine learning algorithms


# In[17]:


x=df.drop(labels=['status'],axis=1)
y=df['status']
x.head()#Splitting the dataset into x and y


# In[18]:


x.head()#Displaying x head


# In[19]:


y.head()#Displaying the y head


# # Splitting the data

# In[20]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)#Splitting the dataset into x_train,y_train,x_test,y_test


# # Machine Learning

# # Logistic Regression

# In[21]:


log_reg=LogisticRegression().fit(x_train,y_train)
#Predict on train
train_preds=log_reg.predict(x_train)
#Accuracy on test
print("Model accuracy on train is:",accuracy_score(y_train,train_preds))

#Predict on test
test_preds=log_reg.predict(x_test)
#Accuracy on test
print("Model accuracy on test is:",accuracy_score(y_test,test_preds))
print('-'*50)

#Confusion matrix
print("Confusion matrix train is:",confusion_matrix(y_train,train_preds))
print("Confusion matrix test is:",confusion_matrix(y_test,test_preds))


# # Random Forest

# In[22]:


rf=RandomForestClassifier().fit(x_train,y_train)
#Predict on train
train_preds2=rf.predict(x_train)
#Accuracy on train
print("Model accuracy on train is:",accuracy_score(y_train,train_preds2))

#Predict on test
test_preds2=rf.predict(x_test)
#Accuracy on test
print("Model accuracy on test is:",accuracy_score(y_test,test_preds2))

#Confusion matrix
print("Confusion matrix train is:",confusion_matrix(y_train,train_preds2))
print("Confusion matrix test is:",confusion_matrix(y_test,test_preds2))


# In[23]:


#Wrong prediction made
print((y_test!=test_preds2).sum(),'/',((y_test==test_preds2).sum()+(y_test!=test_preds2).sum()))


# In[24]:


#Kappa score
print("Kappascore is:",metrics.cohen_kappa_score(y_test,test_preds2))


# In[25]:


#Let us go ahead to compare the predicted and actual value


# In[26]:


test_preds2


# In[27]:


test_preds2,y_test


# In[28]:


#Saving the actual and predicted value to a dataframe


# In[29]:


df1=pd.DataFrame(data=[test_preds2,y_test])


# In[30]:


df1


# In[31]:


df1.T


# In[32]:


#Above 0 means predicted value and 1 is True value


# In[33]:


#Random forest model perform better compared to other models


# In[34]:


#Random forest model gives us an accuracy of 94 percent compared to logistic regression which gives us 84 percent accuracy


# # Applying other Machine learning models to see if there is any improvement in accuracy

# In[35]:


from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# # Decision Tree

# In[36]:


#Fit the model on train data
DT=DecisionTreeClassifier().fit(x,y)

#Predict on train
train_preds3=DT.predict(x_train)
#Accuracy on train
print("Model accuracy on train is:",accuracy_score(y_train,train_preds3))

#Predict on test
test_preds3=DT.predict(x_test)
#Accuracy on test
print("Model accuracy on test is:",accuracy_score(y_test,test_preds3))


# In[37]:


#Confusion matrix
print("Confusion matrix train is:",confusion_matrix(y_train,train_preds3))
print("Confusion matrix test is:",confusion_matrix(y_test,test_preds3))
print("wrong prediction out of total")
print('-'*50)

#Wrong predictions made
print((y_test!=test_preds3).sum(),'/',((y_test==test_preds3).sum()+(y_test!=test_preds3).sum()))
print('-'*50)


# In[38]:


#Kappa score
print("Kappascore is:",metrics.cohen_kappa_score(y_test,test_preds3))


# # Naive Bayes Classifier

# In[40]:


NB=GaussianNB()
NB.fit(x_train,y_train)


# In[41]:


#Fit the model on train data
NB=GaussianNB()
NB.fit(x_train,y_train)

#Predict on train
train_preds4=NB.predict(x_train)
#Accuracy on train
print("Model accuracy on train is:",accuracy_score(y_train,train_preds4))

#Predict on test
test_preds4=NB.predict(x_test)
#Accuracy on test
print("Model accuracy on test is:",accuracy_score(y_test,test_preds4))


# In[42]:


#Confusion matrix
print("Confusion matrix train is:",confusion_matrix(y_train,train_preds4))
print("Confusion matrix test is:",confusion_matrix(y_test,test_preds4))
print("wrong prediction out of total")
print('-'*50)

#Wrong predictions made
print((y_test!=test_preds4).sum(),'/',((y_test==test_preds4).sum()+(y_test!=test_preds4).sum()))
print('-'*50)


# # K-Nearest Neighbors

# In[44]:


#Fit the model on train data
KNN=KNeighborsClassifier().fit(x_train,y_train)

#Predict on train
train_preds5=KNN.predict(x_train)
#Accuracy on train
print("Model accuracy on train is:",accuracy_score(y_train,train_preds5))

#Predict on test
test_preds5=KNN.predict(x_test)
#Accuracy on test
print("Model accuracy on test is:",accuracy_score(y_test,test_preds5))


# In[45]:


#Confusion matrix
print("Confusion matrix train is:",confusion_matrix(y_train,train_preds5))
print("Confusion matrix test is:",confusion_matrix(y_test,test_preds5))
print("wrong prediction out of total")
print('-'*50)

#Wrong predictions made
print((y_test!=test_preds5).sum(),'/',((y_test==test_preds5).sum()+(y_test!=test_preds5).sum()))
print('-'*50)


# # Support Vector Machine

# In[46]:


#Fit the model on train data
SVM=SVC(kernel='linear')
SVM.fit(x_train,y_train)

#Predict on train
train_preds6=SVM.predict(x_train)
#Accuracy on train
print("Model accuracy on train is:",accuracy_score(y_train,train_preds6))

#Predict on test
test_preds6=SVM.predict(x_test)
#Accuracy on test
print("Model accuracy on test is:",accuracy_score(y_test,test_preds6))


# In[48]:


#Confusion matrix
print("Confusion matrix train is:",confusion_matrix(y_train,train_preds6))
print("Confusion matrix test is:",confusion_matrix(y_test,test_preds6))
print("wrong prediction out of total")
print('-'*50)

print("recall",metrics.recall_score(y_test,test_preds6))
print('-'*50)

#Wrong predictions made
print((y_test!=test_preds6).sum(),'/',((y_test==test_preds6).sum()+(y_test!=test_preds6).sum()))
print('-'*50)


# In[49]:


#Kappa score
print("Kappascore is:",metrics.cohen_kappa_score(y_test,test_preds6))


# In[ ]:




