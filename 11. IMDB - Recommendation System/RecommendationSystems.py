#!/usr/bin/env python
# coding: utf-8

# # Recommendation Systems
# 
# Örnek uygulama: Movie Recommendation Software
# 
# Kursumuzun bu bölümünde Recommendation Systems konusu ile ilgili bir movie recommendation sistemi tasarlayacağız.
# 
# <IMG src="https://miro.medium.com/max/1132/1*N0-ikjPv4RUVvS-6KCgLPg.jpeg" width="500" height="500" >

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('users.data', sep='\t', names=column_names)


# In[3]:


df.head()


# In[4]:


# Kaç kayıt varmış görelim:

len(df)


# ### Şimdi diğer dosyamızı yükleyelim,

# In[5]:



movie_titles = pd.read_csv("movie_id_titles.csv")
movie_titles.head()


# In[6]:


# Kaç kayıt varmış görelim:

len(movie_titles)


# In[7]:



df = pd.merge(df, movie_titles, on='item_id')
df.head()


# ### Recommendation Sistemimizi Kuruyoruz:
# 

# In[8]:


# Öncelikle Excel'deki pivot tablo benzeri bir yapı kuruyoruz.
# Bu yapıya göre her satır bir kullanıcı olacak şekilde (yani dataframe'imizin index'i user_id olacak)
# Sütunlarda film isimleri (yani title sütunu) olacak,
# tablo içinde de rating değerleri olacak şekilde bir dataframe oluşturuyoruz!

moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
moviemat.head()


# In[9]:


type(moviemat)


# ### Amaç: Starwars filmine benzer film önerileri yapmak

# Star Wars (1977) filminin user ratinglerine bakalım:

# In[10]:


starwars_user_ratings = moviemat['Star Wars (1977)']
starwars_user_ratings.head()


# corrwith() metodunu kullanarak Star wars filmi ile korelasyonları hesaplatalım:

# In[11]:


similar_to_starwars = moviemat.corrwith(starwars_user_ratings)


# In[12]:


similar_to_starwars


# In[13]:


type(similar_to_starwars)


# #### Bazı kayıtlarda boşluklar olduğu için hata veriyor similar_to_starwars bir seri, biz bunu corr_starwars isimli bir dataframe'e dönüştürelim ve NaN kayıtlarını temizleyip bakalım:

# In[14]:


corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()


# ### Elde ettiğimiz dataframe'i sıralayım ve görelim bakalım star Wars'a en yakın tavsiye edeceği film neymiş:

# In[15]:


corr_starwars.sort_values('Correlation',ascending=False).head(10)


# #### Görüldüğü gibi alakasız sonuçlar çıktı, bu konuyu biraz araştırınca bunun nedeninin bu filmlerin çok az oy alması nedeniyle olduğunu bulacaksınız.. Bu durumu düzeltmek için 100'den az oy alan filmleri eleyelim.. Bu amaçla ratings isimli bir dataframe oluşturalım ve burada her fimin kaç tane oy aldığını (yani oy sayısını) tutalım...
# 

# In[16]:


df.head()


# timestamp sütununa ihtiyacımız yok silelim...

# In[17]:


df.drop(['timestamp'], axis = 1)


# In[19]:


# Her filmin ortalama (mean value) rating değerini bulalım 
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())

# Büyükten küçüğe sıralayıp bakalım...
ratings.sort_values('rating',ascending=False).head()


# #### Dikkat: Bu ortalamalar hesaplanırken kaç oy aldığına bakmadık o yüzden böyle hiç bilmediğimiz filmler çıktı..

# In[20]:


# Şimdi her filmin aldığı oy sayısını bulalım..
ratings['rating_oy_sayisi'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()


# In[21]:


# Şimdi en çok oy alan filmleri büyükten küçüğe sıralayıp bakalım...
ratings.sort_values('rating_oy_sayisi',ascending=False).head()


# In[ ]:


# Tekrar esas amacımıza dönelim ve corr_starwars dataframe'imize rating_oy_sayisi sütununu ekleyelim (join ile) 


# In[23]:


corr_starwars.sort_values('Correlation',ascending=False).head(10)


# In[24]:


corr_starwars = corr_starwars.join(ratings['rating_oy_sayisi'])
corr_starwars.head()


# ### Veee sonuç:

# In[25]:


corr_starwars[corr_starwars['rating_oy_sayisi']>100].sort_values('Correlation',ascending=False).head()


# In[ ]:





# In[ ]:




