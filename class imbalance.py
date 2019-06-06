#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split


# In[2]:


import os
os.chdir('C:\\Analytics\\MachineLearning\\class imbalance')
data = pd.read_csv('creditcard.csv')


# In[3]:


data.columns


# In[4]:


y = data['Class'].values
X = data.drop(['Class','Time'],axis=1).values


# In[6]:


num_neg = (y==0).sum()
num_pos = (y==1).sum()


# In[9]:


# Scaling
scaler = RobustScaler()
X = scaler.fit_transform(X)

# splitting into train-test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt
print(data.groupby('Class').size())
sns.countplot(x='Class',data=data)
plt.show()


# In[16]:


# lets perform simple logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_decision_regions, plot_confusion_matrix

lr = LogisticRegression()

# Fit
lr.fit(X_train,y_train)
# Predict
y_pred = lr.predict(X_test)

# Evaluate
print(classification_report(y_test,y_pred))
plot_confusion_matrix(confusion_matrix(y_test,y_pred))


# In[17]:


lr = LogisticRegression(class_weight='balanced')

# Fit
lr.fit(X_train,y_train)

#predict
y_pred = lr.predict(X_test)

# Evaluate
print(classification_report(y_test,y_pred))
plot_confusion_matrix(confusion_matrix(y_test,y_pred))


# In[18]:


from sklearn.model_selection import GridSearchCV

weights = np.linspace(0.05,0.95,20)

gsc = GridSearchCV(
          estimator = LogisticRegression(),
          param_grid = {
              'class_weight':[{0:x, 1: 1.0-x} for x in weights]
          },
          scoring = 'f1',
          cv=3
      )

grid_result = gsc.fit(X,y)

print('Best parameters : %s' % grid_result.best_params_)

# plot the weight vs f1 score

dataz = pd.DataFrame({'score':grid_result.cv_results_['mean_test_score'],
                     'weight':weights})

dataz.plot(x='weight')


# In[21]:


from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

pipe = make_pipeline(
         SMOTE(),
         LogisticRegression()
)
# Fit
pipe.fit(X_train,y_train)

# Predict.
y_predict = pipe.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
plot_confusion_matrix(confusion_matrix(y_test,y_pred))


# In[25]:


import warnings

pipe = make_pipeline(
       SMOTE(),
       LogisticRegression())

weights = np.linspace(0.005, 0.05, 10)

gsc = GridSearchCV(
         estimator=pipe,
        param_grid={
            'smote__ratio' : weights
        },
          scoring = 'f1',
          cv = 3)

grid_result = gsc.fit(X,y)

print("Best parameters : %s" % grid_result.best_params_)

# Plot the weights vs f1 score
dataz = pd.DataFrame({ 'score': grid_result.cv_results_['mean_test_score'],
                       'weight': weights })
dataz.plot(x='weight')


# In[26]:


pipe = make_pipeline(
      SMOTE(ratio=0.010),
      LogisticRegression())

# Fit
pipe.fit(X_train,y_train)

# Predict
y_predict = pipe.predict(X_test)

# Ecaluate the model
print(classification_report(y_test,y_pred))
plot_confusion_matrix(confusion_matrix(y_test,y_pred))


# In[ ]:




