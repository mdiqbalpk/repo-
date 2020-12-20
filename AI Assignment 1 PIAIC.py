#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
arr = np.array([1, 2, 3, 4])
print(arr[0])


# In[2]:


import numpy as np
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.concatenate((arr1, arr2))
print(arr)


# In[3]:


import numpy as np
array1d = np.array([1, 2, 3, 4, 5, 6])
array2d = np.array([[1, 2, 3], [4, 5, 6]])
array3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(array1d)

print("-" * 10)
print(array2d)
print("-" * 10)
print(array3d)


# In[4]:


import numpy as np
a=np.array([1,2,3])
print(a)


# In[5]:



import numpy as np
 
thearray = np.array([1, 2, 3, 4, 5, 6, 7, 8])
thearray.resize(4)
print(thearray)
 
print("-" * 10)
thearray = np.array([1, 2, 3, 4, 5, 6, 7, 8])
thearray.resize(2, 4)
print(thearray)
 
print("-" * 10)
thearray = np.array([1, 2, 3, 4, 5, 6, 7, 8])
thearray.resize(3, 3)
print(thearray)


# In[6]:


#shape of numpy arr
import numpy as np
 
array1d = np.array([1, 2, 3, 4, 5, 6])
array2d = np.array([[1, 2, 3], [4, 5, 6]])
array3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
 
print(array1d.shape)
print(array2d.shape)
print(array3d.shape)


# In[7]:



# resize array
import numpy as np
 
thearray = np.array([1, 2, 3, 4, 5, 6, 7, 8])
thearray.resize(4)
print(thearray)
 
print("-" * 10)
thearray = np.array([1, 2, 3, 4, 5, 6, 7, 8])
thearray.resize(2, 4)
print(thearray)
 
print("-" * 10)
thearray = np.array([1, 2, 3, 4, 5, 6, 7, 8])
thearray.resize(3, 3)
print(thearray)


# In[8]:


#reshape array
import numpy as np
 
thearray = np.array([1, 2, 3, 4, 5, 6, 7, 8])
thearray = thearray.reshape(2, 4)
print(thearray)
 
print("-" * 10)
thearray = thearray.reshape(4, 2)
print(thearray)
 
print("-" * 10)
thearray = thearray.reshape(8, 1)
print(thearray)


# In[9]:


#indexing
import numpy as np
array1d = np.array([1, 2, 3, 4, 5, 6]) 
print(array1d[-1])   
print("-" * 10)
array2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(array2d)
print("-" * 10)
print(array2d[0, 0])   # Get first row first col


# In[10]:


#zero array
import numpy as np
array1d = np.zeros(3)
print(array1d)
 
array2d = np.zeros((2, 4))
print(array2d)


# In[11]:


#aarange func
import numpy as np
array1d = np.arange(5)  
print(array1d)
array1d = np.arange(0, 12, 2)  
print(array1d)
array2d = np.arange(0, 12, 2).reshape(2, 3)   
print(array2d)


# In[12]:



#random num arr
import numpy as np
print(np.random.rand(3, 2))


# In[13]:


import numpy as np
print(np.identity(3))


# In[14]:


import numpy as np
print(np.diag(np.arange(0, 8, 2)))


# In[16]:


import numpy as np
print(np.identity(3))


# In[17]:


#set union
import numpy as np
 
array1 = np.array([[10, 20, 30], [14, 24, 36]])
array2 = np.array([[20, 40, 50], [24, 34, 46]])
 
print(np.union1d(array1, array2))


# In[18]:



#set intersection
import numpy as np
 
array1 = np.array([[10, 20, 30], [14, 24, 36]])
array2 = np.array([[20, 40, 50], [24, 34, 46]])
print(np.intersect1d(array1, array2))


# In[19]:



#set difference
import numpy as np
 
array1 = np.array([[10, 20, 30], [14, 24, 36]])
array2 = np.array([[20, 40, 50], [24, 34, 46]])
print(np.setdiff1d(array1, array2))


# In[20]:



import numpy as np
 
before = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]])
choices = [5, 10, 15]
 
after = np.choose(before, choices)
print(after)
 
print("-" * 10)
 
before = np.array([[0, 0, 0], [2, 2, 2], [1, 1, 1]])
choice1 = [5, 10, 15]
choice2 = [8, 16, 24]
choice3 = [9, 18, 27]
 
after = np.choose(before, (choice1, choice2, choice3))
print(after)


# In[21]:


import numpy as np
 
array1 = np.array([[10, 20, 30], [40, 50, 60]])
 
print(np.sin(array1))


# In[22]:


import numpy as np
 
array1 = np.array([[10, 20, 30], [40, 50, 60]])
print(np.cos(array1))


# In[23]:



import numpy as np
 
array1 = np.array([[10, 20, 30], [40, 50, 60]])
print(np.tan(array1))


# In[24]:


import numpy as np
array1 = np.array([[10, 20, 30], [40, 50, 60]])
print(np.sqrt(array1))


# In[25]:


import numpy as np
array1 = np.array([[10, 20, 30], [40, 50, 60]])
print(np.exp(array1))


# In[26]:



import numpy as np
array1 = np.array([[10, 20, 30], [40, 50, 60]])
print(np.log10(array1))


# In[27]:



import numpy as np
array3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(array3d)
 
print(array3d[0, 0, 0])
print(array3d[0, 0, 1])
print(array3d[0, 0, 2])


# In[28]:


import numpy as np
 
type1 = np.array([1, 2, 3, 4, 5, 6])
type2 = np.array([1.5, 2.5, 0.5, 6])
type3 = np.array(['a', 'b', 'c'])
type4 = np.array(["Canada", "Australia"], dtype='U5')
type5 = np.array([555, 666], dtype=float)
 
print(type1.dtype)
print(type2.dtype)
print(type3.dtype)
print(type4.dtype)
print(type5.dtype)


# In[29]:


import numpy as np
 
print(np.linspace(1, 10))  
print(np.linspace(1, 10, 7, endpoint=False))


# In[30]:


import numpy as np
x = np.array(42)
print("x: ", x)
print("The type of x: ", type(x))
print("The dimension of x:", np.ndim(x))


# In[31]:


import numpy as np

E = np.ones((2,3))
print(E)

F = np.ones((3,4),dtype=int)
print(F)


# In[32]:


import numpy as np

x = np.array([[42,22,12],[44,53,66]], order='F')
y = x.copy()

x[0,0] = 1001
print(x)

print(y)


# In[33]:



import numpy as np
a = np.array([3,8,12,18,7,11,30])
odd_elements = a[1::2]
print(odd_elements)


# In[34]:


import numpy as np

num1 = 4
num2 = 6

x = np.lcm(num1, num2)

print(x)


# In[35]:


import numpy as np

arr = np.array([1, 2, 3, 4])

x = np.prod(arr)

print(x)


# In[36]:


import numpy as np
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])

newarr = np.prod([arr1, arr2], axis=1)

print(newarr)


# In[37]:


import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6])

newarr = np.array_split(arr, 3)

print(newarr)


# In[38]:


import numpy as np

arr = np.array([3, 2, 0, 1])

print(np.sort(arr))


# In[39]:


import numpy as np
 
type1 = np.array([1, 2, 3, 4, 5, 6])
type2 = np.array([1.5, 2.5, 0.5, 6])
type3 = np.array(['a', 'b', 'c'])
type4 = np.array(["Canada", "Australia"], dtype='U5')
type5 = np.array([555, 666], dtype=float)
 
print(type1.dtype)
print(type2.dtype)
print(type3.dtype)
print(type4.dtype)
print(type5.dtype)


# In[40]:


import numpy as np
x = np.array(42)
print("x: ", x)
print("The type of x: ", type(x))
print("The dimension of x:", np.ndim(x))


# In[41]:


# resize array
import numpy as np
 
thearray = np.array([1, 2, 3, 4, 5, 6, 7, 8])
thearray.resize(4)
print(thearray)
 
print("-" * 10)
thearray = np.array([1, 2, 3, 4, 5, 6, 7, 8])
thearray.resize(2, 4)
print(thearray)
 
print("-" * 10)
thearray = np.array([1, 2, 3, 4, 5, 6, 7, 8])
thearray.resize(3, 3)
print(thearray)


# In[42]:


import numpy as np
 
array1 = np.array([[10, 20, 30], [40, 50, 60]])
print(np.cos(array1))


# In[43]:


import numpy as np

x = np.array([[42,22,12],[44,53,66]], order='F')
y = x.copy()

x[0,0] = 1001
print(x)

print(y)


# In[44]:


import numpy as np
 
array1 = np.array([[10, 20, 30], [40, 50, 60]])
 
print(np.sin(array1))


# In[45]:


import numpy as np
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.concatenate((arr1, arr2))
print(arr)


# In[46]:


import numpy as np
arr = np.array([1, 2, 3, 4])
print(arr[0])


# In[47]:


import numpy as np
 
array1 = np.array([[10, 20, 30], [40, 50, 60]])
print(np.cos(array1))


# In[48]:


import numpy as np
array1d = np.array([1, 2, 3, 4, 5, 6])
array2d = np.array([[1, 2, 3], [4, 5, 6]])
array3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(array1d)

print("-" * 10)
print(array2d)
print("-" * 10)
print(array3d)


# In[49]:


#random num arr
import numpy as np
print(np.random.rand(3, 2))


# In[50]:


#set union
import numpy as np
 
array1 = np.array([[10, 20, 30], [14, 24, 36]])
array2 = np.array([[20, 40, 50], [24, 34, 46]])
 
print(np.union1d(array1, array2))


# In[ ]:




