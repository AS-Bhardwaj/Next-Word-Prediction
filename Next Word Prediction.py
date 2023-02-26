#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import the necessary libraries

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential 
from keras.layers import Dense , Bidirectional , LSTM , Embedding
from keras.utils.data_utils import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
import pickle
import re
import string as str1


# In[3]:


#get the data 
#Next,let's explore our data.

data = pd.read_csv("medium_data.csv")
data.head()


# In[4]:


#By analyzing the text data present in data['title'] column , i found that there are some special characters like ['“','”','\xa0', '\u200a'] so we are going to replace them by 'nothing' or whitespaces. 

data['title']=data['title'].apply(lambda x : x.replace("“" , "") ).apply(lambda x : x.replace("”" , ""))
data['title'] = data['title'].apply(lambda x: x.replace(u'\xa0',u' '))
data['title'] = data['title'].apply(lambda x: x.replace('\u200a',' '))


# In[5]:


#Let's have a look at our title column again.
data['title']


# In[ ]:


#it looks fine


# In[ ]:


#so next we are going to tokenize eacch word, which means we will provide a no. for each unique word.
#this is necessary as machines don't uderstand charcters.  


# In[8]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['title'])
wordindex = tokenizer.word_index
totalwords = len(wordindex)+1
totalwords


# In[ ]:


#ok, now we have got total unique words in our title column and we have provided each word with a token which is nothing but integer 


# In[26]:


#next we will convert the complete titles in sequences of the tokens which we have given to each word above.
#We will convert the sentences into list of tokens in which each tken represents a word
#for e.g "How to Use ggplot2 in Python " --> [20, 522, 1031, 4432, 12, 2342]. this just an e.g 

#then in the 2nd loop we are creating the dataset for our model.
#for e.g x1=[20,522] , y1=[1031] ; x2=[20,522,1031] , y2=[4432] ; x3=[20,522,1031,4432] , y3=[12]


# In[31]:


sequence=[]
input_sequence=[]
for line in data['title']:
    #print([line])
    seq= tokenizer.texts_to_sequences([line])[0]
    sequence.append(seq)
    for i in range(1 , len(seq)):
        x = seq[:i+1]
        input_sequence.append(x)
        


# In[27]:


#let's have a look what our title coulumn looks when converted into sequence

sequence


# In[29]:


#this is how next word prediction models are trained, now we just need to pad these sequences to make them of same length.
input_sequence


# In[12]:


#As the Dense Layer accepts inputs of fixed length we have to convert all the title sequences to same length.
#We will do this by padding all the titles with zeros to make them of same length.
#for this we need to know the what is the max length of title in data['title'] column
#Bcz we are going to make all sentences of that length only , as we can increase the lengths of other titles but can't reduce them.

len(input_sequence)
maxlen = max(len(i) for i in input_sequence)


# In[13]:


#we are using the pad_sequences library of keras in which we have provided padding = 'pre'.
#which means that we are adding zeros in left of the sequences and padding='post' means adding zeros to the right side.

X = np.array(pad_sequences(input_sequence , maxlen , padding='pre' ))


# In[14]:


#let's have a look at our padded sequnces

X


# In[17]:


#you know what we are going to do now?
#we will extract input and output arrays from X.
#our input will be X[:,:-1] which means all the rows but just excluding the last column and we will tae that last column as output.

inputseq=np.array(input_sequence)
xs = X[:,:-1]
labels = X[:,-1]


# In[18]:


#let's have a look at our input array

xs


# In[19]:


#now look at our labels.

labels


# In[ ]:


#wait.
#don't you think there is something incomplete.
#oh yess , we haven't converted our labels into one hot encodings.
#so , now we will convert our labels into one hot encodings using the to_categoical function of keras.


# In[20]:


ys = to_categorical(labels)


# In[21]:


ys.shape


# In[22]:


xs.shape


# In[23]:


totalwords


# In[24]:


#let's prepare the model architecture.
#we will use embedding layer for dimensionality reduction and text undersanding.
#we will convert each word into vector of 100 dimension using embedding layer. this will provide the model an understaniding of the context.
#Then we arre using Bidirectional LSTM Layer which is followed by Dense Layer which comprises of Softmax Activation.


model = Sequential()
model.add(Embedding(input_dim = totalwords , output_dim=100 , input_length=maxlen-1))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(totalwords , activation='Softmax'))
model.compile(optimizer='adam' , loss='categorical_crossentropy' , metrics=['accuracy'])


# In[25]:


#finally fitting the data in our model and training it.
#Let's go.

model.fit(xs , ys , epochs=40)


# In[ ]:





# In[ ]:





# In[ ]:




