#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml


# In[2]:


mnist = fetch_openml('mnist_784' , as_frame=False,parser="auto")
#The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. The database is also widely used for training and testing in the field of machine learning.


# In[3]:


mnist # Test = 10000,Train = 60000(Total = 70000)


# In[4]:


x, y = mnist["data"],mnist["target"] # here x = image and y = label


# In[5]:


x.shape # An array consumes less memory and is convenient to use. NumPy uses much less memory to store data and it provides a mechanism of specifying the data types.


# In[6]:


y.shape


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


import matplotlib
import matplotlib.pyplot as plt


# In[31]:


some_digit = x[3629]
some_digit_image = some_digit.reshape(28, 28) # lets reshape it to plot it


# In[32]:


plt.imshow(some_digit_image,  cmap=matplotlib.cm.binary , interpolation ="nearest")
# cmap-color map used to specify color   
# interpolation-algorithm used to blend square colors; with 'nearest' colors will not be blendedd
plt.axis("off")


# In[33]:


y[3601] #  To get label


# In[34]:


x_train,x_test = x[0:6000] ,x[6000:7000] # mnist has already splitted so no need of splitting


# In[35]:


y_train,y_test = y[0:6000] ,y[6000:7000] 


# In[36]:


import numpy as np 
shuffle_index = np.random.permutation(6000) # numpy.random.permutation(x) Return the random sequence of permuted values.
x_train ,y_train = x_train[shuffle_index], y_train[shuffle_index] # shuffling to avoid better result 


# # Creating a 2 detector

# In[37]:


y_train = y_train.astype(np.int8) # Change into integer
y_test = y_test.astype(np.int8)  
y_train_2 = (y_train == 2)
y_test_2 = (y_test ==2)


# In[38]:


y_train # for binary classifier 


# In[39]:


from sklearn.linear_model import LogisticRegression 
# Regression is a technique for investigating the relationship between independent variables or features and a dependent variable or outcome.
# Logistic regression is used to handle the classification problems. Linear regression provides a continuous output but Logistic regression provides discreet output


# In[40]:


clf = LogisticRegression(tol = 0.1 , solver ="lbfgs") # tol-tolerance for getting run fast


# In[41]:


clf.fit(x_train, y_train_2)


# In[42]:


clf.predict([some_digit])


# # Cross Validation

# In[43]:


from sklearn.model_selection import cross_val_score # a technique for evaluating ML models by training several ML models on subsets of the available input data and evaluating them on the complementary subset of the data
a = cross_val_score(clf,x_train,y_train_2,cv=3 ,scoring = "accuracy") # cv - The number of splits to use,scoring - The error metric to use


# In[46]:


a.mean()


# In[47]:


# Accuraaacy is not the metric of classifier

#In[48]:

from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(clf,x_train,y_train_2,cv=3)

#In[49]:
y_train_pred # Prediction by classifier


#Calculating confusion matrix

#In[50]:
from sklearn.metrics import confusion_matrix

#In[51]:
confusion_matrix(y_train_2 ,y_train_pred)

#In[52]:
confusion_matrix(y_train_2 ,y_train_2) # When  we get perfect prediction  


#In[53]:
Precision and Recall

#In[54]:
from sklearn.metrics import precision_score ,recall_score

#In[55]:
precision_score(y_train_2 ,y_train_pred)  # This is my precision score

#In[56]:
recall_score(y_train_2 ,y_train_pred) # This is my recall score 


#F1 - Score

#In[57]:
from sklearn.metrics import f1_score

#In[58]:
f1_score(y_train_2 ,y_train_pred)

#Precision Recall Curve

#In[59]:
from sklearn.metrics import precision_recall_curve

#In[60]:
y_scores = cross_val_predict(clf,x_train,y_train_2,cv=3 ,method = "decision_function")

#In[61]:
y_scores

#In[62]:
precisions, recalls, thresholds = precision_recall_curve(y_train_2, y_scores)

#In[63]:
precisions

#In[64]:
recalls

#In[65]:
thresholds #  Logistic regression with the sigmoid returns a probability between 0 and 1 for an input data sample to belong to the positive class. A probability of 0.99 means that the email is very likely to be spam, and a probability of 0.003 that it is very likely to be non-spam. If the probability is 0.51, the classifier is less able immediately to determine the nature of the email. A value above that threshold indicates "spam"; a value below indicates "not spam." It is tempting to assume that the classification threshold should always be 0.5, but thresholds are problem-dependent, and are therefore values that you must tune.


#Plotting the Precision Recall Curve

#In[66]:
plt.plot(thresholds, precisions[:-1],"b--", label = "Precision")
plt.plot(thresholds,  recalls[:-1], "g-", label = "Recall")
plt.xlabel("Thresholds")
plt.legend(loc = "upper left") # For writing the labels precisions , recalls
plt.ylim([0,1])
plt.show()


