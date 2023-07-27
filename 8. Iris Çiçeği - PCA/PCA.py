#!/usr/bin/env python
# coding: utf-8

# # PCA -  Principal Component Analysis
# 
# 
# 
# Bu dersimizde örnek olarak kullanacagimiz veri seti yapay ögrenme alaninin en popüler veri setlerinden “Iris” veri seti. Iris veri seti 3 Iris bitki türüne (Iris setosa, Iris virginica ve Iris versicolor) ait, her bir türden 50 örnek olmak üzere toplam 150 örnek sayisina sahip bir veri setidir. Her bir örnek için 4 özellik tanimlanmistir: taç yaprak uzunlugu, taç yaprak genisligi, çanak yaprak genisligi, çanak yaprak uzunluğu('sepal length','sepal width','petal length','petal width'). 
# 
# Veri setimizde, her bir bitki örnegi ayri bir gözlemi (örnegi) ifade ederken; bitki tür ismi bagimli(dependent) degisken, bitkilerin ölçülen 4 temel özelligi ise bagimsiz(independent) degiskenleri ifade eder.

# In[1]:



import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

url = "pca_iris.data"
# datasetimizi Pandas DataFrame içine yüklüyoruz..
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])

df


# In[2]:


features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# feature'ları x olarak ayıralım:
x = df[features]

# target'i y olarak ayıralım:
y = df[['target']]


# #### Değerleri Scale etmemiz gerekiyor. Çünkü her bir feature çok farklı boyutlarda ve bunların yapay zeka tarafından eşit ağırlıklarda dengelenmesi gerekiyor. Bu amaçla standart scaler  kullanarak tüm verileri mean = 0 and variance = 1 olacak şekilde değiştiriyoruz.

# In[3]:


# Standardizing the features
x = StandardScaler().fit_transform(x)


# In[4]:


# Bakalım scale etmiş mi?
x


# ### PCA Projection 4 boyuttan - 2 boyuta
# 
# Orjinal verilerimiz 4 boyuta sahip: 'sepal length', 'sepal width', 'petal length', 'petal width'
# 
# Biz PCA yaparak bunları 2 boyuta indirgeyeceğiz ancak şunu belirtmeliyim ki PCA indirgeme işlemi sonucunda elde edeceğimiz 2 boyutun herhangi bir anlam ifade etmeyen başlıklara sahip olacak.. Yani 4 feature'dan 2 tanesini basit bir şekilde atmak değil yaptığımız..
# 
# 

# In[5]:


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])


# In[6]:


principalDf


# ### Şimdi en son target sütunumuzu da PCA dataframe'imizin sonuna ekleyelim:

# In[7]:


final_dataframe = pd.concat([principalDf, df[['target']]], axis = 1)


# In[8]:


final_dataframe.head()


# In[ ]:





# ### Son olarak da final dataframe'imizi görselleştirip bakalım:

# Basit bir çizim yapalım:

# In[9]:


dfsetosa= final_dataframe[df.target=='Iris-setosa']
dfvirginica = final_dataframe[df.target=='Iris-virginica']
dfversicolor = final_dataframe[df.target=='Iris-versicolor']
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')

plt.scatter(dfsetosa['principal component 1'], dfsetosa['principal component 2'],color='green')
plt.scatter(dfvirginica['principal component 1'], dfvirginica['principal component 2'],color='red')
plt.scatter(dfversicolor['principal component 1'], dfversicolor['principal component 2'],color='blue')


# ### Daha profesyonel bir plotting yapalım:

# In[10]:


targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['g', 'b', 'r']

plt.xlabel('principal component 1')
plt.ylabel('principal component 2')

for target, col in zip(targets,colors):
    dftemp = final_dataframe[df.target==target]
    plt.scatter(dftemp['principal component 1'], dftemp['principal component 2'], color=col)


# In[ ]:





# In[11]:


pca.explained_variance_ratio_


# In[12]:


pca.explained_variance_ratio_.sum()


# In[ ]:




