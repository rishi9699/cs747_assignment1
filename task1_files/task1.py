import numpy as np
import matplotlib.pyplot as plt

def do_eps_greedy(arms, randomSeed, epsilon, horizon):
    lenn=len(arms)
    cumu_reward=np.zeros(horizon)
    rng = np.random.default_rng(randomSeed)
    pulls_till_now = np.zeros(lenn, dtype='uint32')
    empirical_means = np.zeros(lenn)
    
    i=0
    while i<horizon:
        if (rng.binomial(1, epsilon))==1:
            arm_index = rng.integers(lenn)
        else:
            arm_index = np.argmax(empirical_means)
            
        curr_pull = rng.binomial(1, arms[arm_index])
        cumu_reward[i]=cumu_reward[i-1]+curr_pull
        empirical_means[arm_index] = (empirical_means[arm_index] * pulls_till_now[arm_index] + curr_pull)/(pulls_till_now[arm_index]+1)
        pulls_till_now[arm_index]+=1
        i+=1
    #print(empirical_means)
    return cumu_reward


def do_ucb(arms, randomSeed, scale, horizon):

    curr_rew = 0
    cumu_reward=np.zeros(horizon)
    rng = np.random.default_rng(randomSeed)
    lenn = len(arms)
    empirical_means = np.zeros(lenn)
    
    i=0
    
    while i<lenn:
        curr_rew = rng.binomial(1, arms[i])
        cumu_reward[i] = cumu_reward[i-1] + curr_rew
        empirical_means[i] = curr_rew
        i+=1
    
    pulls_till_now = np.ones(lenn, dtype='uint32')
    
    while i<horizon:
        ucb = empirical_means + np.sqrt((scale*np.log(i))/pulls_till_now)
        arm_index = np.argmax(ucb)
        
        curr_rew = rng.binomial(1, arms[arm_index])
        cumu_reward[i] = cumu_reward[i-1] + curr_rew
        empirical_means[arm_index] = (empirical_means[arm_index] * pulls_till_now[arm_index] + curr_rew)/(pulls_till_now[arm_index]+1)
        pulls_till_now[arm_index]+=1
    
        i+=1
    #print(pulls_till_now)    
    return cumu_reward

def do_thompson(arms, randomSeed, horizon):
    cumu_reward=np.zeros(horizon)
    rng = np.random.default_rng(randomSeed)
    lenn = len(arms)
    successesp1 = np.ones(lenn, dtype='uint32')
    failuresp1 = np.ones(lenn, dtype='uint32')
    
    i=0
    while i<horizon:
        arm_index = np.argmax(rng.beta(successesp1, failuresp1))
        if (rng.binomial(1, arms[arm_index]))==1:
            successesp1[arm_index]+=1
            cumu_reward[i] = cumu_reward[i-1] + 1
        else:
            failuresp1[arm_index]+=1
            cumu_reward[i] = cumu_reward[i-1]
        i+=1
    
    #print((successesp1-1)/(successesp1+failuresp1-2))
    return cumu_reward
    
def do_kl_ucb(arms, randomSeed, horizon):
    cumu_reward=np.zeros(horizon)
    lenn=len(arms)
    c=3
    rng = np.random.default_rng(randomSeed)
    empirical_means = np.zeros(lenn)
    pulls_till_now = np.zeros(lenn, dtype='uint32')
    
    i=0
    while i<max(4, lenn):
        curr_rew = rng.binomial(1, arms[i%lenn])
        cumu_reward[i] = cumu_reward[i-1] + curr_rew
        empirical_means[i%lenn] = (empirical_means[i%lenn] * pulls_till_now[i%lenn] + curr_rew)/(pulls_till_now[i%lenn]+1)
        pulls_till_now[i%lenn]+=1
        i+=1
        
    kl_ucbs = np.zeros(lenn)
    while i<horizon:
        rhs_arms = (np.log(i) + c * np.log(np.log(i)))/pulls_till_now
        
        delta = 0.0001
        # BEGIN CALCULATION OF q's
        for z in range(lenn):
            rhs = rhs_arms[z]
            p = empirical_means[z]
            
            if p==1:
                q = 1
                
            elif p==0:
                left = p
                right = 1
                
                q_candidate = (p+1)/2
                kldiv = np.log(1/(1-q_candidate))
                
                while abs(kldiv-rhs)>delta:
                    if kldiv<rhs:
                        left = q_candidate
                    else:
                        right = q_candidate
                    q_candidate = (left+right)/2
                    kldiv = np.log(1/(1-q_candidate))
                    
                q = q_candidate
                
            else:
                left = p
                right = 1
                
                q_candidate = (p+1)/2
                kldiv = p*np.log(p/q_candidate) + (1-p)*np.log((1-p)/(1-q_candidate))
                
                while abs(kldiv-rhs)>delta:
                    if kldiv<rhs:
                        left = q_candidate
                    else:
                        right = q_candidate
                    q_candidate = (left+right)/2
                    kldiv = p*np.log(p/q_candidate) + (1-p)*np.log((1-p)/(1-q_candidate))
                    
                q = q_candidate
                
            kl_ucbs[z] = q
            
        arm_index = np.argmax(kl_ucbs)
        curr_rew = rng.binomial(1, arms[arm_index])
        cumu_reward[i] = cumu_reward[i-1] + curr_rew
        empirical_means[arm_index] = (empirical_means[arm_index] * pulls_till_now[arm_index] + curr_rew)/(pulls_till_now[arm_index]+1)
        pulls_till_now[arm_index]+=1
        i+=1
        
    return cumu_reward

#%%instance 1 eps greedy
arms = [0.4, 0.8]
sums_rews = np.zeros(102400)

horizons = [100, 400, 1600, 6400, 25600, 102400]

for rs in range(50):
    f=open('task1_instance1_epsgreedy.txt', 'a')
    cum_rews = do_eps_greedy(arms, rs, 0.02, 102400)
    for j in horizons:
        REG = (j*np.max(arms) - cum_rews[j-1])
        string_to_write = '../instances/instances-task1/i-1.txt, epsilon-greedy-t1, '+str(rs)+', 0.02, 2, 0, '+str(j)+', '+str(round(REG, 2))+', 0\n'
        _ = f.write(string_to_write)
    f.close()
    sums_rews+=cum_rews
    print(rs)
sums_rews/=50
regs = (np.max(arms)*np.arange(1, 102401, 1)) - sums_rews
np.save('t1-i1-epsgreedy', regs)

#%%instance 1 ucb
arms = [0.4, 0.8]
sums_rews = np.zeros(102400)

horizons = [100, 400, 1600, 6400, 25600, 102400]

for rs in range(50):
    f=open('task1_instance1_ucb.txt', 'a')
    cum_rews = do_ucb(arms, rs, 2, 102400)
    for j in horizons:
        REG = (j*np.max(arms) - cum_rews[j-1])
        string_to_write = '../instances/instances-task1/i-1.txt, ucb-t1, '+str(rs)+', 0.02, 2, 0, '+str(j)+', '+str(round(REG, 2))+', 0\n'
        _ = f.write(string_to_write)
    f.close()
    sums_rews+=cum_rews
    print(rs)
sums_rews/=50
regs = (np.max(arms)*np.arange(1, 102401, 1)) - sums_rews
np.save('t1-i1-ucb', regs)

#%%instance 1 thompson
arms = [0.4, 0.8]
sums_rews = np.zeros(102400)

horizons = [100, 400, 1600, 6400, 25600, 102400]

for rs in range(50):
    f=open('task1_instance1_thompson.txt', 'a')
    cum_rews = do_thompson(arms, rs, 102400)
    for j in horizons:
        REG = (j*np.max(arms) - cum_rews[j-1])
        string_to_write = '../instances/instances-task1/i-1.txt, thompson-sampling-t1, '+str(rs)+', 0.02, 2, 0, '+str(j)+', '+str(round(REG, 2))+', 0\n'
        _ = f.write(string_to_write)
    f.close()
    sums_rews+=cum_rews
    print(rs)
sums_rews/=50
regs = (np.max(arms)*np.arange(1, 102401, 1)) - sums_rews
np.save('t1-i1-thompson', regs)

#%%instance 1 kl_ucb
arms = [0.4, 0.8]
sums_rews = np.zeros(102400)

horizons = [100, 400, 1600, 6400, 25600, 102400]

for rs in range(50):
    f=open('task1_instance1_klucb.txt', 'a')
    cum_rews = do_kl_ucb(arms, rs, 102400)
    for j in horizons:
        REG = (j*np.max(arms) - cum_rews[j-1])
        string_to_write = '../instances/instances-task1/i-1.txt, kl-ucb-t1, '+str(rs)+', 0.02, 2, 0, '+str(j)+', '+str(round(REG, 2))+', 0\n'
        _ = f.write(string_to_write)
    f.close()
    sums_rews+=cum_rews
    print(rs)
sums_rews/=50
regs = (np.max(arms)*np.arange(1, 102401, 1)) - sums_rews
np.save('t1-i1-klucb', regs)

#%%instance 2 eps greedy
arms = [0.4, 0.3, 0.5, 0.2, 0.1]
sums_rews = np.zeros(102400)

horizons = [100, 400, 1600, 6400, 25600, 102400]

for rs in range(50):
    f=open('task1_instance2_epsgreedy.txt', 'a')
    cum_rews = do_eps_greedy(arms, rs, 0.02, 102400)
    for j in horizons:
        REG = (j*np.max(arms) - cum_rews[j-1])
        string_to_write = '../instances/instances-task1/i-2.txt, epsilon-greedy-t1, '+str(rs)+', 0.02, 2, 0, '+str(j)+', '+str(round(REG, 2))+', 0\n'
        _ = f.write(string_to_write)
    f.close()
    sums_rews+=cum_rews
    print(rs)
sums_rews/=50
regs = (np.max(arms)*np.arange(1, 102401, 1)) - sums_rews
np.save('t1-i2-epsgreedy', regs)
#%%instance 2 ucb
arms = [0.4, 0.3, 0.5, 0.2, 0.1]
sums_rews = np.zeros(102400)

horizons = [100, 400, 1600, 6400, 25600, 102400]

for rs in range(50):
    f=open('task1_instance2_ucb.txt', 'a')
    cum_rews = do_ucb(arms, rs, 2, 102400)
    for j in horizons:
        REG = (j*np.max(arms) - cum_rews[j-1])
        string_to_write = '../instances/instances-task1/i-2.txt, ucb-t1, '+str(rs)+', 0.02, 2, 0, '+str(j)+', '+str(round(REG, 2))+', 0\n'
        _ = f.write(string_to_write)
    f.close()
    sums_rews+=cum_rews
    print(rs)
sums_rews/=50
regs = (np.max(arms)*np.arange(1, 102401, 1)) - sums_rews
np.save('t1-i2-ucb', regs)
#%%instance 2 thompson
arms = [0.4, 0.3, 0.5, 0.2, 0.1]
sums_rews = np.zeros(102400)

horizons = [100, 400, 1600, 6400, 25600, 102400]

for rs in range(50):
    f=open('task1_instance2_thompson.txt', 'a')
    cum_rews = do_thompson(arms, rs, 102400)
    for j in horizons:
        REG = (j*np.max(arms) - cum_rews[j-1])
        string_to_write = '../instances/instances-task1/i-2.txt, thompson-sampling-t1, '+str(rs)+', 0.02, 2, 0, '+str(j)+', '+str(round(REG, 2))+', 0\n'
        _ = f.write(string_to_write)
    f.close()
    sums_rews+=cum_rews
    print(rs)
sums_rews/=50
regs = (np.max(arms)*np.arange(1, 102401, 1)) - sums_rews
np.save('t1-i2-thompson', regs)
#%%instance 2 kl_ucb
arms = [0.4, 0.3, 0.5, 0.2, 0.1]
sums_rews = np.zeros(102400)

horizons = [100, 400, 1600, 6400, 25600, 102400]

for rs in range(50):
    f=open('task1_instance2_klucb.txt', 'a')
    cum_rews = do_kl_ucb(arms, rs, 102400)
    for j in horizons:
        REG = (j*np.max(arms) - cum_rews[j-1])
        string_to_write = '../instances/instances-task1/i-2.txt, kl-ucb-t1, '+str(rs)+', 0.02, 2, 0, '+str(j)+', '+str(round(REG, 2))+', 0\n'
        _ = f.write(string_to_write)
    f.close()
    sums_rews+=cum_rews
    print(rs)
sums_rews/=50
regs = (np.max(arms)*np.arange(1, 102401, 1)) - sums_rews
np.save('t1-i2-klucb', regs)
#%%instance 3 eps greedy
arms = [0.15, 0.23, 0.37, 0.44, 0.5, 0.32, 0.78, 0.21, 0.82, 0.56, 0.34, 0.56, 0.84, 0.76, 0.43, 0.65, 0.73, 0.92, 0.10, 0.89, 0.48, 0.96, 0.60, 0.54, 0.49]
sums_rews = np.zeros(102400)

horizons = [100, 400, 1600, 6400, 25600, 102400]

for rs in range(50):
    f=open('task1_instance3_epsgreedy.txt', 'a')
    cum_rews = do_eps_greedy(arms, rs, 0.02, 102400)
    for j in horizons:
        REG = (j*np.max(arms) - cum_rews[j-1])
        string_to_write = '../instances/instances-task1/i-3.txt, epsilon-greedy-t1, '+str(rs)+', 0.02, 2, 0, '+str(j)+', '+str(round(REG, 2))+', 0\n'
        _ = f.write(string_to_write)
    f.close()
    sums_rews+=cum_rews
    print(rs)
sums_rews/=50
regs = (np.max(arms)*np.arange(1, 102401, 1)) - sums_rews
np.save('t1-i3-epsgreedy', regs)
#%%instance 3 ucb
arms = [0.15, 0.23, 0.37, 0.44, 0.5, 0.32, 0.78, 0.21, 0.82, 0.56, 0.34, 0.56, 0.84, 0.76, 0.43, 0.65, 0.73, 0.92, 0.10, 0.89, 0.48, 0.96, 0.60, 0.54, 0.49]
sums_rews = np.zeros(102400)

horizons = [100, 400, 1600, 6400, 25600, 102400]

for rs in range(50):
    f=open('task1_instance3_ucb.txt', 'a')
    cum_rews = do_ucb(arms, rs, 2, 102400)
    for j in horizons:
        REG = (j*np.max(arms) - cum_rews[j-1])
        string_to_write = '../instances/instances-task1/i-3.txt, ucb-t1, '+str(rs)+', 0.02, 2, 0, '+str(j)+', '+str(round(REG, 2))+', 0\n'
        _ = f.write(string_to_write)
    f.close()
    sums_rews+=cum_rews
    print(rs)
sums_rews/=50
regs = (np.max(arms)*np.arange(1, 102401, 1)) - sums_rews
np.save('t1-i3-ucb', regs)
#%%instance 3 thompson
arms = [0.15, 0.23, 0.37, 0.44, 0.5, 0.32, 0.78, 0.21, 0.82, 0.56, 0.34, 0.56, 0.84, 0.76, 0.43, 0.65, 0.73, 0.92, 0.10, 0.89, 0.48, 0.96, 0.60, 0.54, 0.49]
sums_rews = np.zeros(102400)

horizons = [100, 400, 1600, 6400, 25600, 102400]

for rs in range(50):
    f=open('task1_instance3_thompson.txt', 'a')
    cum_rews = do_thompson(arms, rs, 102400)
    for j in horizons:
        REG = (j*np.max(arms) - cum_rews[j-1])
        string_to_write = '../instances/instances-task1/i-3.txt, thompson-sampling-t1, '+str(rs)+', 0.02, 2, 0, '+str(j)+', '+str(round(REG, 2))+', 0\n'
        _ = f.write(string_to_write)
    f.close()
    sums_rews+=cum_rews
    print(rs)
sums_rews/=50
regs = (np.max(arms)*np.arange(1, 102401, 1)) - sums_rews
np.save('t1-i3-thompson', regs)
#%%instance 3 kl_ucb
arms = [0.15, 0.23, 0.37, 0.44, 0.5, 0.32, 0.78, 0.21, 0.82, 0.56, 0.34, 0.56, 0.84, 0.76, 0.43, 0.65, 0.73, 0.92, 0.10, 0.89, 0.48, 0.96, 0.60, 0.54, 0.49]
sums_rews = np.zeros(102400)

horizons = [100, 400, 1600, 6400, 25600, 102400]

for rs in range(50):
    f=open('task1_instance3_klucb.txt', 'a')
    cum_rews = do_kl_ucb(arms, rs, 102400)
    for j in horizons:
        REG = (j*np.max(arms) - cum_rews[j-1])
        string_to_write = '../instances/instances-task1/i-3.txt, kl-ucb-t1, '+str(rs)+', 0.02, 2, 0, '+str(j)+', '+str(round(REG, 2))+', 0\n'
        _ = f.write(string_to_write)
    f.close()
    sums_rews+=cum_rews
    print(rs)
sums_rews/=50
regs = (np.max(arms)*np.arange(1, 102401, 1)) - sums_rews
np.save('t1-i3-klucb', regs)