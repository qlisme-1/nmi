from turtle import left, right
import gym
import numpy as np
import time
import sys
from yolo import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl



env = gym.make('GridWorld-v1')


num_episodes = 5000# 共进行5000场游戏
max_number_of_steps = 10# 每场游戏最大步数

# 以栈的方式记录成绩
goal_average_steps = 100 # 平均分                     
num_consecutive_iterations = 100 # 栈的容量
last_time_steps = np.zeros(num_consecutive_iterations)  # 只存储最近100场的得分（可以理解为是一个容量为100的栈）

env = gym.make('GridWorld-v1')
pos = [
    [0,1,2,3],
    [4,5,6,7],
    [8,9,10,11],
    [12,13,14,15],
]
transfer = dict()
size = 4
for i in range(size, size * size):  # 上、下、左、右各方向寻路
    transfer[str(i) + '_0'] = i - 4 # 向上
for i in range(size * (size - 1)):  
    transfer[str(i) + '_1'] = i + 4 # 向下
for i in range(1, size * size):
    if i % size == 0:
        continue
    transfer[str(i) + '_2'] = i - 1 # 向左
for i in range(size * size):
    if (i + 1) % size == 0:
        continue
    transfer[str(i) + '_3'] = i + 1 # 向右


q_table = np.random.uniform(low=-1, high=1, size=(4 * 4, 4)) #创建q_table为矩阵形式
# 根据本次的行动及其反馈（下一个时间步的状态），返回下一次的最佳行动
# epsilon_coefficient为贪心策略中的ε，取值范围[0,1]，取值越大，行为越随机
# 当epsilon_coefficient取值为0时，将完全按照q_table行动。故可作为训练模型与运用模型的开关值。
def get_action(state, action, observation, reward, episode, epsilon_coefficient=0.0):
    # print(observation)
    next_state = observation #由observation确定接下来的状态
    epsilon = epsilon_coefficient * (0.99 ** episode)  # ε-贪心策略中的ε
    if epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[next_state]) #选取qtable下一个状态中分数最高的动作
    else:
        next_action = np.random.choice([0, 1, 2, 3])
    # -------------------------------------训练学习，更新q_table----------------------------------
    alpha = 0.2  # 学习系数α
    gamma = 0.99  # 报酬衰减系数γ
    q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (
            reward + gamma * q_table[next_state, next_action]) #根据reward确定下一个动作的qtable赋值
    # -------------------------------------------------------------------------------------------
    return next_action, next_state

timer = time.time()
yolo = YOLO()

episodes=[]
rewards=[]

for episode in range(num_episodes):
    env.reset()  # 初始化本场游戏的环境
    episode_reward = 0  # 初始化本场游戏的得分
    q_table_cache = q_table # 创建q_table还原点，如若训练次数超次，则不作本次训练记录。
    v_state = 3
    state = env.state  #状态是环境中获得的
    action = np.argmax(q_table[state]) #动作根据q_table来选取的
    virtual_word = np.zeros([4, 4])  #创建虚拟世界
    
    for t in range(max_number_of_steps):
        #time.sleep(0.1)
        image = env.render(mode='rgb_array')    # 更新并渲染游戏画面
        plt.imsave('./figu/1.jpg',image)
        img = './figu/1.jpg'
        image = Image.open(img)
        predicted_classes,boxes = yolo.detect_image(image,crop = False, count=False)
        
        print(predicted_classes)
        print(boxes)
        #print(rois)
        #print(id)

        #print(int(c))
        #print(box)
        
        for index in range(0,5):
            x = int((boxes[index][0]-100)/100)
            y = int((boxes[index][1]-100)/100)
            if predicted_classes[index]=='robot':
                v_state = pos[x][y]
            if predicted_classes[index]=='obstacle':
                virtual_word[x][y] = -100
            if predicted_classes[index]=='destination':
                virtual_word[x][y] = 100
                
        #print(v_state)
        virtual_word[int(v_state/4)][int(v_state%4)] = -20 #走过的路赋负值，避免重复走过去的路
        print(virtual_word)
        flag = 0
        a_score = np.zeros([4])  #创建得分矩阵，初始状态均为0，存储评判某一状态下上下左右四个动作的得分
        while flag<=size:
            flag = flag + 1
            # print("v_state   ",end="")
            # print(v_state)
            # print("action   ", end="")
            # print(action)
            key = "%d_%d" % (v_state, action)   #v_state为3(在终点或障碍物的位置）或pos[x][y]（在机器人的位置）
            if key not in transfer:
                a_score[action] = -100  #未在transfer之内说明这个动作为跑出棋盘的动作，所以赋值为-100
                get_action(v_state, action, v_state, -100, 0.5)
                action = np.argmax(q_table[v_state])
                continue
            else:
                next_state = transfer[key] #0-15
                if virtual_word[int(next_state/4)][int(next_state%4)]<0:  
                    a_score[action] = virtual_word[int(next_state/4)][int(next_state%4)] #动作的得分为虚拟世界的数值
                    get_action(v_state,action,next_state,virtual_word[int(next_state/4)][int(next_state%4)],0.5)
                    action = np.argmax(q_table[v_state])
                    continue
                else:
                    break
        if flag>size:
            action = np.argmax(a_score) #选出得分最大的动作
        # action = np.random.choice([0, 1, 2, 3])  # 随机决定小车运动的方向
        observation, reward, done, info = env.step(action)  # 进行活动,并获取本次行动的反馈结果
        action, state = get_action(state, action, observation, reward, episode, 0.5)  # 作出下一次行动的决策
        episode_reward += reward
        if done:
            np.savetxt("q_table.txt", q_table, delimiter=",")
            print('已完成 %d 次训练，本次训练共进行 %d 步数。episode_reward：%d，平均分： %f' % (episode, t + 1, reward, last_time_steps.mean()))
            last_time_steps = np.hstack((last_time_steps[1:], [reward]))    # 更新最近100场游戏的得分stack
            break      
    q_table = q_table_cache # 超次还原q_table

    episode_reward = -100
    print('已完成 %d 次训练，本次训练共进行 %d 步数。episode_reward：%d，平均分： %f' % (episode, t + 1, reward, last_time_steps.mean()))

    rewards.append(last_time_steps.mean())
    episodes.append(episode)


    last_time_steps = np.hstack((last_time_steps[1:], [reward]))    # 更新最近100场游戏的得分stack
    
    if (last_time_steps.mean() >= goal_average_steps):
        np.savetxt("q_table.txt", q_table, delimiter=",")
        print('用时 %d s,训练 %d 次后，模型到达测试标准!' % (time.time() - timer, episode))

        fig,ax = plt.subplots()
        ax.plot(episodes,rewards,linewidth=3)
        ax.set_xlabel("episode",fontsize=14)
        ax.set_ylabel("reward",fontsize=14)
        plt.show() 
        env.close()

        sys.exit()



env.close()
sys.exit()

