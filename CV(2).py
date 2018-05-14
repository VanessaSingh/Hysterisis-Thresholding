
# coding: utf-8

# In[17]:


import matplotlib.pyplot as plt
from skimage import data, filters
import numpy as np


# In[18]:


fig, ax = plt.subplots(nrows=2, ncols=2,
                       subplot_kw={'adjustable': 'box-forced'})


# In[19]:


#from PIL import Image
#from numpy import*
#image=asarray(Image.open('/home/vanessa/Downloads/doggo.jpg'))


# In[20]:


#print(image.shape[0])
#print(image.shape[1])
#print(image.shape[2])


# In[21]:


#x=image.shape[0]
#y=image.shape[1]*image.shape[2]

#image.resize((x,y))


# In[22]:


image = data.coins()
edges = filters.sobel(image)


# In[23]:


low = 0.1
high = 0.35


# In[24]:


from skimage import feature


# In[25]:


lowt = (edges > low).astype(int)
hight = (edges > high).astype(int)
hyst = filters.apply_hysteresis_threshold(edges, low, high)


# In[26]:


ax[0, 0].imshow(image, cmap='gray')
#plt.show()
ax[0, 0].set_title('Original image')


# In[27]:


ax[0, 1].imshow(edges, cmap='magma')
#plt.show()
ax[0, 1].set_title('Sobel edges')


# In[28]:


ax[1, 0].imshow(lowt, cmap='magma')
ax[1, 0].set_title('Low threshold')


# In[29]:


ax[1, 1].imshow(hight + hyst, cmap='magma')
ax[1, 1].set_title('Hysteresis threshold')


# In[30]:


for a in ax.ravel():
    a.axis('off')


# In[31]:


plt.tight_layout()


# In[32]:


plt.show()

