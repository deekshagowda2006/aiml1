#!/usr/bin/env python
# coding: utf-8

# In[36]:


import nltk


# In[37]:


file=open("ammu.txt")
text=file.read()
text


# In[38]:


from nltk.tokenize import sent_tokenize 
sentence=sent_tokenize(text)
print("no of sentences:",len(sentence))
for i in range(len(sentence)):
    print("Sentence",i+1,":\n",sentence[i])


# In[39]:


from nltk.tokenize import word_tokenize 
words=word_tokenize(text)
print("no of words:",len(words))
print(words)


# In[40]:


from nltk.probability import FreqDist 
freqdisk=FreqDist(words)
freqdisk


# In[41]:


import matplotlib.pyplot as plt 
import pandas as pd 
freqdisk=pd.Series(dict(freqdisk))
fig,ax=plt.subplots(figsize=(8,8))
freqdisk.plot(kind='bar')
plt.ylabel('count')
plt.show()


# In[42]:


text=text.upper()
text


# In[43]:


text=text.lower()
text


# In[44]:


import re
text=re.sub('[^A-Za-z0-9]+', ' ', text)
text


# In[45]:


text=re.sub('[\s*\d\s*]', ' ', text).strip()
text


# In[46]:


stopwords=nltk.corpus.stopwords.words('english')
re_words=[]
for word in words:
    if word in stopwords:
        pass
    else:
        re_words.append(word)


# In[47]:


freqdisk=FreqDist(re_words)
freqdisk=pd.Series(dict(freqdisk))
fig,ax=plt.subplots(figsize=(8,8))
freqdisk.plot(kind='bar')
plt.ylabel('count')
plt.show()


# In[48]:


from wordcloud import WordCloud,STOPWORDS
stopwords=set(STOPWORDS)
wordcloud=WordCloud(width=800,height=800,background_color='white',stopwords=stopwords,min_font_size=10).generate(text)
plt.figure(figsize=(10,10),facecolor=None)
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[49]:


from skimage.io import imread
cloud=imread('ac.png')
plt.imshow(cloud)


# In[50]:


from wordcloud import WordCloud,STOPWORDS
stopwords=set(STOPWORDS)
wordcloud=WordCloud(width=800,height=800,background_color='white',stopwords=stopwords,min_font_size=10,mask=cloud).generate(text)
plt.figure(figsize=(10,10),facecolor=None)
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[51]:


from nltk.metrics.distance import edit_distance 
nltk.download('words')
from nltk.corpus import words 
correct_words=words.words()
incorrect_words='happpy','edokation'
for word in incorrect_words:
    temp=[(edit_distance(word,w),w) for w in correct_words if w[0]==word[0]]
    print(sorted(temp,key=lambda val:val[0])[0][1])


# In[55]:


from nltk.tokenize import word_tokenize 
words=word_tokenize(text)
from nltk.stem import PorterStemmer 
ps=PorterStemmer()
stem_sent=[ps.stem(words_sent) for words_sent in words]
print(stem_sent)


# In[56]:


from nltk.stem.wordnet import WordNetLemmatizer
wl=WordNetLemmatizer()
lem_sent=[ps.stem(words_sent) for words_sent in words]
print(stem_sent)


# In[58]:


words=word_tokenize(text)
print("pos",nltk.pos_tag(words))


# In[73]:


df=pd.read_csv('Iris.csv')
df


# In[74]:


df.fillna(df.mean(),inplace=True)


# In[75]:


df.isnull().sum()


# In[76]:


x=df.iloc[ :, 1:3].values
x


# In[77]:


from sklearn.cluster import KMeans 
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("elbow method")
plt.xlabel('no of clusters')
plt.xlabel('wcss')
plt.show()


# In[83]:


km1=KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_means=km1.fit_predict(x)
y_means


# In[84]:


plt.scatter(x[y_means==0,0],x[y_means==0,1],s=100,c='pink',label='c1')
plt.scatter(x[y_means==1,0],x[y_means==1,1],s=100,c='green',label='c2')
plt.scatter(x[y_means==2,0],x[y_means==2,1],s=100,c='brown',label='c2')
plt.scatter(km1.cluster_centers_[:,0],km1.cluster_centers_[:,1],s=100,c='blue',label='centroid')
plt.title("clusters")
plt.xlabel("SepalLengthCm")
plt.ylabel('SepalWidthCm')
plt.legend()
plt.show()



# In[ ]:




