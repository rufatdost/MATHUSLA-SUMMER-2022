#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as math


# In[3]:


data = pd.read_csv('DELAY - TEST 4 - 15 MIN 700 MM.csv', low_memory=False)
data


# In[4]:


data.columns


# In[5]:


delay = np.array(data['Delay(1R-2R)'])


# In[6]:


argument = np.where(delay == max(delay))


# In[13]:


def histogramCalc(data, small_length, length): 
    data.columns
    delay = np.array(data['Delay(1R-2R)'])
    argument = np.where(delay == max(delay))
    a = np.delete(delay, argument)
    plt.xlabel("DELAY. TEST 4 - 15 MIN - JUST FIBER- 700 MM")
    plt.ylabel("Frequency")
    x = np.random.gamma(4, 0.5, 1000)
    plt.axvline(a.mean(),color='k', linestyle='dashed', linewidth=1)
    plt.hist(a, bins = 5000)

    plt.show()

    mean_rounded = "{:.3f}".format((np.mean(a)) * (10**9))
    sr_rounded = "{:.3f}".format((np.std(a)) * (10**9))
    median_rounded = "{:.3f}".format((np.median(a)) * (10**9))
    
    
    mean = "Mean: " + str(mean_rounded) + " ns"
    SD = "SD: " + str(sr_rounded) + " ns"
    median = "Median: " + str(median_rounded) + " ns"
    
    print(mean)
    print(SD)
    print(median)
    
    distance = length - 2*(small_length)

    spofl = 299792458
    velocity = ((distance/1000) / (np.mean(a)))
    percentage = (velocity / spofl) * 100 
    
    print(str(percentage) + " %")


# In[14]:


data = pd.read_csv('DELAY - TEST 8 - 12 hrs.csv', low_memory=False)

histogramCalc(data, 200, 2000)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




