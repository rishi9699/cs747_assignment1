#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
def do_task4(instances, randomSeed, threshold, horizon):
    arms = instances.shape[0]
    r_support = np.array([0, 0.25, 0.5, 0.75, 1])
    threshold_index = np.where(r_support>=threshold)[0][0]
    rng = np.random.default_rng(randomSeed)
    pulls_till_now = np.zeros(arms, dtype='uint32')
    thresh_means = np.zeros(arms)
    good_pulls = np.zeros(arms, dtype='uint32')

    epsilon=0.1 #make this argument optional in the function and the real code
    
    i=0
    cumu_highs = np.zeros(horizon, dtype='uint32')
    while i<horizon:
        if (rng.binomial(1, epsilon))==1:
            arm_index = rng.integers(arms)
        else:
            arm_index = np.argmax(thresh_means)
            
        rew_obt = rng.choice(r_support, p=instances[arm_index])
        
        curr_high=0
        if (np.where(r_support == rew_obt)[0][0])>=threshold_index:
            good_pulls[arm_index]+=1
            curr_high = 1
            
        cumu_highs[i] = cumu_highs[i-1] + curr_high
        pulls_till_now[arm_index]+=1
        thresh_means[arm_index] = good_pulls[arm_index]/pulls_till_now[arm_index]
        
        i+=1
    
    #print(pulls_till_now)
    #print(cumu_highs[-1])
    return cumu_highs
    #return (horizon*np.max(np.sum(instances*r_support, axis=1)) - cumu_reward[-1])
    

#%%
#instance 1 t0.2
threshold = 0.2
r_support = np.array([0, 0.25, 0.5, 0.75, 1])
threshold_index = np.where(r_support>=threshold)[0][0]
arms = 3
instances = np.zeros((arms, len(r_support)))
instances[0] = [0, 0.6, 0, 0.3, 0.1]
instances[1] = [0, 0, 0.8, 0.1, 0.1]
instances[2] = [0.2, 0.2, 0.3, 0.2, 0.1]
horizon = 102400


sums_cumu_highs = np.zeros(102400)
horizons = [100, 400, 1600, 6400, 25600, 102400]

for rs in range(50):
    f=open('task4_instance1_t0p2.txt', 'a')
    cumu_highs = do_task4(instances, rs, 0.2, 102400)
    for j in horizons:
        string_to_write = '../instances/instances-task4/i-1.txt, alg-t4, '+str(rs)+', 0.02, 2, 0.2, '+str(j)+', 0, '+str(int(cumu_highs[j-1]))+'\n'
        _ = f.write(string_to_write)
    f.close()
    sums_cumu_highs+=cumu_highs
    print(rs)
sums_cumu_highs/=50
highs_regret = 1*np.arange(1, 102401, 1) - sums_cumu_highs    #check this for every instance, threshold
np.save('t4-i1-0p2', highs_regret)
#%%
#instance 1 t0.6
threshold = 0.6
r_support = np.array([0, 0.25, 0.5, 0.75, 1])
threshold_index = np.where(r_support>=threshold)[0][0]
arms = 3
instances = np.zeros((arms, len(r_support)))
instances[0] = [0, 0.6, 0, 0.3, 0.1]
instances[1] = [0, 0, 0.8, 0.1, 0.1]
instances[2] = [0.2, 0.2, 0.3, 0.2, 0.1]
horizon = 102400


sums_cumu_highs = np.zeros(102400)
horizons = [100, 400, 1600, 6400, 25600, 102400]

for rs in range(50):
    f=open('task4_instance1_t0p6.txt', 'a')
    cumu_highs = do_task4(instances, rs, 0.6, 102400)
    for j in horizons:
        string_to_write = '../instances/instances-task4/i-1.txt, alg-t4, '+str(rs)+', 0.02, 2, 0.6, '+str(j)+', 0, '+str(int(cumu_highs[j-1]))+'\n'
        _ = f.write(string_to_write)
    f.close()
    sums_cumu_highs+=cumu_highs
    print(rs)
sums_cumu_highs/=50
highs_regret = 0.4*np.arange(1, 102401, 1) - sums_cumu_highs    #check this for every instance, threshold
np.save('t4-i1-0p6', highs_regret)

#%%
#instance 2 t0.2
threshold = 0.2
r_support = np.array([0, 0.25, 0.5, 0.75, 1])
threshold_index = np.where(r_support>=threshold)[0][0]
arms = 3
instances = np.zeros((arms, len(r_support)))
instances[0] = [0.15, 0.15, 0.4, 0.1, 0.2]
instances[1] = [0.1, 0.2, 0.17, 0.43, 0.1]
instances[2] = [0.19, 0.41, 0.15, 0.15, 0.1]
horizon = 102400


sums_cumu_highs = np.zeros(102400)
horizons = [100, 400, 1600, 6400, 25600, 102400]

for rs in range(50):
    f=open('task4_instance2_t0p2.txt', 'a')
    cumu_highs = do_task4(instances, rs, 0.2, 102400)
    for j in horizons:
        string_to_write = '../instances/instances-task4/i-2.txt, alg-t4, '+str(rs)+', 0.02, 2, 0.2, '+str(j)+', 0, '+str(int(cumu_highs[j-1]))+'\n'
        _ = f.write(string_to_write)
    f.close()
    sums_cumu_highs+=cumu_highs
    print(rs)
sums_cumu_highs/=50
highs_regret = 0.9*np.arange(1, 102401, 1) - sums_cumu_highs    #check this for every instance, threshold
np.save('t4-i2-0p2', highs_regret)

#%%
#instance 2 t0.6
threshold = 0.6
r_support = np.array([0, 0.25, 0.5, 0.75, 1])
threshold_index = np.where(r_support>=threshold)[0][0]
arms = 3
instances = np.zeros((arms, len(r_support)))
instances[0] = [0.15, 0.15, 0.4, 0.1, 0.2]
instances[1] = [0.1, 0.2, 0.17, 0.43, 0.1]
instances[2] = [0.19, 0.41, 0.15, 0.15, 0.1]
horizon = 102400


sums_cumu_highs = np.zeros(102400)
horizons = [100, 400, 1600, 6400, 25600, 102400]

for rs in range(50):
    f=open('task4_instance2_t0p6.txt', 'a')
    cumu_highs = do_task4(instances, rs, 0.6, 102400)
    for j in horizons:
        string_to_write = '../instances/instances-task4/i-2.txt, alg-t4, '+str(rs)+', 0.02, 2, 0.6, '+str(j)+', 0, '+str(int(cumu_highs[j-1]))+'\n'
        _ = f.write(string_to_write)
    f.close()
    sums_cumu_highs+=cumu_highs
    print(rs)
sums_cumu_highs/=50
highs_regret = 0.53*np.arange(1, 102401, 1) - sums_cumu_highs    #check this for every instance, threshold
np.save('t4-i2-0p6', highs_regret)
