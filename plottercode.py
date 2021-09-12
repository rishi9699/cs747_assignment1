# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 16:12:52 2021

@author: Rishi
"""
import matplotlib.pyplot as plt
import numpy as np
ppp1 = np.load('t2-i1.npy')
ppp2 = np.load('t2-i2.npy')
ppp3 = np.load('t2-i3.npy')
ppp4 = np.load('t2-i4.npy')
ppp5 = np.load('t2-i5.npy')




#%%
plt.plot(np.arange(0.02, 0.32, 0.02), ppp1)
plt.plot(np.arange(0.02, 0.32, 0.02), ppp2)
plt.plot(np.arange(0.02, 0.32, 0.02), ppp3)
plt.plot(np.arange(0.02, 0.32, 0.02), ppp4)
plt.plot(np.arange(0.02, 0.32, 0.02), ppp5)
plt.xlabel('c')
plt.ylabel('Regret')
plt.legend(['i-1', 'i-2', 'i-3', 'i-4', 'i-5'])
#plt.xticks(np.arange(0.02, 0.32, 0.02))
#plt.yticks(ppp1)
plt.title('Regret vs. c')

#%%
plt.close()
#%%
pp2 = np.load('t3-i2.npy')
plt.plot(np.arange(0, 102400, 1), pp2)
plt.xscale('log')
plt.xlabel('Horizon')
plt.ylabel('Regret')
plt.legend(['i-2'])
plt.title('Regret vs. Horizon for i-2')
#%%
pppp1 = np.load('t4-i1-0p6.npy')
plt.plot(np.arange(0, 102400, 1), pppp1)
plt.xscale('log')
plt.xlabel('Horizon')
plt.ylabel('Highs-Regret')
plt.legend(['i-2 with threshold 0.6'])
plt.title('Highs-Regret vs. Horizon for i-2 and threshold 0.6')
#%%
epg = np.load('t1-i3-epsgreedy.npy')
ucbb = np.load('t1-i3-ucb.npy')
klucbb= np.load('t1-i3-klucb.npy')
thsn= np.load('t1-i3-thompson.npy')
plt.plot(np.arange(0, 102400, 1), epg)
plt.plot(np.arange(0, 102400, 1), ucbb)
plt.plot(np.arange(0, 102400, 1), klucbb)
plt.plot(np.arange(0, 102400, 1), thsn)
plt.legend(['$\epsilon$G3','UCB','KL-UCB','Thompson Sampling'])
plt.xscale('log')
plt.xlabel('Horizon')
plt.ylabel('Regret')
plt.title('Regret vs. Horizon for i-3')

print(epg[-1])
print(ucbb[-1])
print(klucbb[-1])
print(thsn[-1])