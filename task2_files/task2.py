#%%
import numpy as np
import matplotlib.pyplot as plt

#%%    
def do_ucb(arms, randomSeed, scale, horizon):
    tot_rew = 0
    curr_rew = 0
    rng = np.random.default_rng(randomSeed)
    lenn = len(arms)
    empirical_means = np.zeros(lenn)
    
    i=0
    
    while i<lenn:
        curr_rew = rng.binomial(1, arms[i])
        tot_rew += curr_rew
        empirical_means[i] = curr_rew
        i+=1
    
    pulls_till_now = np.ones(lenn, dtype='uint32')
    
    while i<horizon:
        ucb = empirical_means + np.sqrt((scale*np.log(i))/pulls_till_now)
        arm_index = np.argmax(ucb)
        
        curr_rew = rng.binomial(1, arms[arm_index])
        tot_rew += curr_rew
        empirical_means[arm_index] = (empirical_means[arm_index] * pulls_till_now[arm_index] + curr_rew)/(pulls_till_now[arm_index]+1)
        pulls_till_now[arm_index]+=1
    
        i+=1
    #print(pulls_till_now)    
    return (horizon * np.max(arms) - tot_rew)
        

#%%
#instance1
scales = np.arange(0.02, 0.32, 0.02)

instance1_average_regrets = np.zeros(np.shape(scales))


arms = [0.7, 0.2]


for ind in range(15):
    reg_sum_seeds = 0
    f=open('task2_instance1.txt', 'a')
    for rs in range(50):
        REG = do_ucb(arms, rs, round(scales[ind],2), 10000)
        reg_sum_seeds+=REG
        string_to_write = '../instances/instances-task2/i-1.txt, ucb-t2, '+str(rs)+', 0.02, '+str(round(scales[ind],2))+', 0, 10000, '+str(REG)+', 0\n'
        _ = f.write(string_to_write)
    f.close()
    reg_sum_seeds/=50
    instance1_average_regrets[ind] = reg_sum_seeds
    print(ind)

np.save('t2-i1', instance1_average_regrets)
#%%
#instance2
scales = np.arange(0.02, 0.32, 0.02)
instance2_average_regrets = np.zeros(np.shape(scales))


arms = [0.3, 0.7]

for ind in range(15):
    reg_sum_seeds = 0
    f=open('task2_instance2.txt', 'a')
    for rs in range(50):
        REG = do_ucb(arms, rs, round(scales[ind],2), 10000)
        reg_sum_seeds+=REG
        string_to_write = '../instances/instances-task2/i-2.txt, ucb-t2, '+str(rs)+', 0.02, '+str(round(scales[ind],2))+', 0, 10000, '+str(REG)+', 0\n'
        _ = f.write(string_to_write)
    f.close()
    reg_sum_seeds/=50
    instance2_average_regrets[ind] = reg_sum_seeds
    print(ind)

np.save('t2-i2', instance2_average_regrets)
#%%
#instance3
scales = np.arange(0.02, 0.32, 0.02)
instance3_average_regrets = np.zeros(np.shape(scales))


arms = [0.7, 0.4]

for ind in range(15):
    reg_sum_seeds = 0
    f=open('task2_instance3.txt', 'a')
    for rs in range(50):
        REG = do_ucb(arms, rs, round(scales[ind],2), 10000)
        reg_sum_seeds+=REG
        string_to_write = '../instances/instances-task2/i-3.txt, ucb-t2, '+str(rs)+', 0.02, '+str(round(scales[ind],2))+', 0, 10000, '+str(REG)+', 0\n'
        _ = f.write(string_to_write)
    f.close()
    reg_sum_seeds/=50
    instance3_average_regrets[ind] = reg_sum_seeds
    print(ind)
    
np.save('t2-i3', instance3_average_regrets)

#%%
#instance4
scales = np.arange(0.02, 0.32, 0.02)
instance4_average_regrets = np.zeros(np.shape(scales))


arms = [0.7, 0.5]

for ind in range(15):
    reg_sum_seeds = 0
    f=open('task2_instance4.txt', 'a')
    for rs in range(50):
        REG = do_ucb(arms, rs, round(scales[ind],2), 10000)
        reg_sum_seeds+=REG
        string_to_write = '../instances/instances-task2/i-4.txt, ucb-t2, '+str(rs)+', 0.02, '+str(round(scales[ind],2))+', 0, 10000, '+str(REG)+', 0\n'
        _ = f.write(string_to_write)
    f.close()
    reg_sum_seeds/=50
    instance4_average_regrets[ind] = reg_sum_seeds
    print(ind)
    
np.save('t2-i4', instance4_average_regrets)

#%%
#instance5
scales = np.arange(0.02, 0.32, 0.02)
instance5_average_regrets = np.zeros(np.shape(scales))


arms = [0.6, 0.7]

for ind in range(15):
    reg_sum_seeds = 0
    f=open('task2_instance5.txt', 'a')
    for rs in range(50):
        REG = do_ucb(arms, rs, round(scales[ind],2), 10000)
        reg_sum_seeds+=REG
        string_to_write = '../instances/instances-task2/i-5.txt, ucb-t2, '+str(rs)+', 0.02, '+str(round(scales[ind],2))+', 0, 10000, '+str(REG)+', 0\n'
        _ = f.write(string_to_write)
    f.close()
    reg_sum_seeds/=50
    instance5_average_regrets[ind] = reg_sum_seeds
    print(ind)
    
np.save('t2-i5', instance5_average_regrets)
