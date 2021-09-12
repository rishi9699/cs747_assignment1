#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
def do_task3(instances, randomSeed, horizon):
    arms = instances.shape[0]
    r_support = np.array([0, 0.25, 0.5, 0.75, 1])
    rng = np.random.default_rng(randomSeed)
    pulls_till_now = np.zeros(arms, dtype='uint32')
    empirical_means = np.zeros(arms)
    epsilon=0.1 #make this argument optional in the function and the real code
    
    i=0
    cumu_reward = np.zeros(horizon)
    while i<horizon:
        if (rng.binomial(1, epsilon))==1:
            arm_index = rng.integers(arms)
        else:
            arm_index = np.argmax(empirical_means)
            
        rew_obt = rng.choice(r_support, p=instances[arm_index])
        cumu_reward[i]=cumu_reward[i-1]+rew_obt
        empirical_means[arm_index] = (empirical_means[arm_index] * pulls_till_now[arm_index] + rew_obt)/(pulls_till_now[arm_index]+1)
        pulls_till_now[arm_index]+=1
        i+=1
    
    #print(pulls_till_now)
    return cumu_reward
    #return (horizon*np.max(np.sum(instances*r_support, axis=1)) - cumu_reward[-1])

#%%
arms = 3
instances = np.zeros((arms, 5))
instances[0] = [0, 0.6, 0, 0.3, 0.1]
instances[1] = [0, 0, 0.8, 0.1, 0.1]
instances[2] = [0.2, 0.2, 0.3, 0.2, 0.1]

#instance1
sums_rews = np.zeros(102400)
r_support = np.array([0, 0.25, 0.5, 0.75, 1])
horizons = [100, 400, 1600, 6400, 25600, 102400]
for rs in range(50):
    f=open('task3_instance1.txt', 'a')
    cum_rews = do_task3(instances, rs, 102400)
    for j in horizons:
        REG = (j*np.max(np.sum(instances*r_support, axis=1)) - cum_rews[j-1])
        string_to_write = '../instances/instances-task3/i-1.txt, alg-t3, '+str(rs)+', 0.02, 2, 0, '+str(j)+', '+str(round(REG, 2))+', 0\n'
        _ = f.write(string_to_write)
    f.close()
    sums_rews+=cum_rews
    print(rs)
sums_rews/=50
#print(102400*np.max(np.sum(instances*r_support, axis=1)) - sums_rews[-1])
regs = (np.max(np.sum(instances*r_support, axis=1))*np.arange(1, 102401, 1)) - sums_rews
np.save('t3-i1', regs)
#%%
# instance 2
arms = 3
instances = np.zeros((arms, 5))
instances[0] = [0.15, 0.15, 0.4, 0.1, 0.2]
instances[1] = [0.1, 0.2, 0.17, 0.43, 0.1]
instances[2] = [0.19, 0.41, 0.15, 0.15, 0.1]

sums_rews = np.zeros(102400)
r_support = np.array([0, 0.25, 0.5, 0.75, 1])
horizons = [100, 400, 1600, 6400, 25600, 102400]
for rs in range(50):
    f=open('task3_instance2.txt', 'a')
    cum_rews = do_task3(instances, rs, 102400)
    for j in horizons:
        REG = (j*np.max(np.sum(instances*r_support, axis=1)) - cum_rews[j-1])
        string_to_write = '../instances/instances-task3/i-2.txt, alg-t3, '+str(rs)+', 0.02, 2, 0, '+str(j)+', '+str(round(REG, 2))+', 0\n'
        _ = f.write(string_to_write)
    f.close()
    sums_rews+=cum_rews
    print(rs)
sums_rews/=50
#print(102400*np.max(np.sum(instances*r_support, axis=1)) - sums_rews[-1])
regs = (np.max(np.sum(instances*r_support, axis=1))*np.arange(1, 102401, 1)) - sums_rews
np.save('t3-i2', regs)
