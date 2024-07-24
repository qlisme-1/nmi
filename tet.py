import numpy as np

def param():
    num_episodes = 5000# 共进行5000场游戏
    max_number_of_steps = 100# 每场游戏最大步数
    # 以栈的方式记录成绩
    goal_average_steps = 100 # 平均分
    num_consecutive_iterations = 100 # 栈的容量
    last_time_steps = np.zeros(num_consecutive_iterations)  # 只存储最近100场的得分（可以理解为是一个容量为100的栈）
    q_table = np.random.uniform(low=-1, high=1, size=(10 * 10, 4))
    return num_episodes,max_number_of_steps,goal_average_steps,num_consecutive_iterations,last_time_steps,q_table

def state_trans(siz):
    size = siz
    pos=[]
    for i in range(size):
        for j in range(size):
            pos[i]=[i*10+j]
    transfer = dict()
    
    for i in range(size, size * size):  # 上、下、左、右各方向寻路
        transfer[str(i) + '_0'] = i - size

    for i in range(size * (size - 1)):
        transfer[str(i) + '_1'] = i + size

    for i in range(1, size * size):
        if i % size == 0:
            continue
        transfer[str(i) + '_2'] = i - 1

    for i in range(size * size):
        if (i + 1) % size == 0:
            continue
        transfer[str(i) + '_3'] = i + 1
    return pos,transfer

def get_action(q_table,state, action, observation, reward, episode, epsilon_coefficient=0.2):
            # print(observation)
            next_state = observation
            epsilon = epsilon_coefficient * (0.99 ** episode)  # ε-贪心策略中的ε
            if epsilon <= np.random.uniform(0, 1):
                next_action = np.argmax(q_table[next_state])
            else:
                next_action = np.random.choice([0, 1, 2, 3])
            # -------------------------------------训练学习，更新q_table----------------------------------
            alpha = 0.2  # 学习系数α
            gamma = 0.99  # 报酬衰减系数γ
            q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (
                    reward + gamma * q_table[next_state, next_action])
            # -------------------------------------------------------------------------------------------
            return next_action, next_state,q_table