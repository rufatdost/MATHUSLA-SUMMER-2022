#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as math
import warnings
import scipy
from scipy.stats import norm


# In[67]:


data = pd.read_csv('DELAY - TEST 10 - 12 hrs.csv', low_memory=False)
data


# In[68]:


data.columns


# In[69]:


delay = np.array(data['Delay(1R-2R)'])


# In[70]:


argument = np.where(delay == max(delay))


# In[154]:


def histogramCalc(data, small_length, length, left_value, right_value, pulsar_bias, pulsar_freq): 
    data.columns
    delay = np.array(data['Delay(2R-1R)'])
    
    p = 10**(-8)
    
    delay = delay[(delay > left_value*p) & (delay < (right_value*p))]

    argument = np.where(delay == max(delay))
    
    
    
    
   
    a = np.delete(delay, argument)
    plt.xlabel("TEST 10 - " + str(small_length) + "mm. " +"Bias: " + str(pulsar_bias) +"." +" Freq: " + str(pulsar_freq))
    plt.ylabel("Frequency")
    x = np.random.gamma(4, 0.5, 1000)
    plt.axvline(a.mean(),color='b', linestyle='dashed', linewidth=1)
    
    # Gaussian Fit:
    mu, std = norm.fit(a) 
    
    


    plt.hist(a, bins = "auto",density=True, alpha=0.6, color='b')
    
   # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    

    
    
  
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit Values: {:.2f} and {:.2f}".format(mu, std)
    plt.title(title)
    

    plt.show()

    mean_rounded = "{:.3f}".format((np.mean(a)) * (10**9))
    median_rounded = "{:.3f}".format((np.median(a)) * (10**9))
    
    
    
    
    mean = "Mean: " + str(mean_rounded) + " ns"
    SD = "SD: " + str(np.std(a)) + " ns"
    median = "Median: " + str(median_rounded) + " ns"
    
    print(mean)
    print(SD)
    print(median)
    
    print("SD of the curve:" + str(std))
    print("Mean of the curve:" + str(mu))
    
    distance = length - 2*(small_length)

    spofl = 299792458
    velocity = ((distance/1000) / (np.mean(a)))
    percentage = (velocity / spofl) * 100 
    
    print(str(percentage) + " %")
    
    
    
    
    
    
                      
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




