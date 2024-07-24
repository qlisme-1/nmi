import gym
import numpy as np
import time
import sys
import PIL
from yolo import YOLO
from PIL import Image
from queue import Queue
from queue import Empty
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from os import getcwd
#yolo训练函数
from yolo_train import *
from data_process import image_config
from yolo_train import _yolo_tarin_

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
    transfer[str(i) + '_0'] = i - 4
for i in range(size * (size - 1)):
    transfer[str(i) + '_1'] = i + 4
for i in range(1, size * size):
    if i % size == 0:
        continue
    transfer[str(i) + '_2'] = i - 1
for i in range(size * size):
    if (i + 1) % size == 0:
        continue
    transfer[str(i) + '_3'] = i + 1


q_table = np.random.uniform(low=-1, high=1, size=(4 * 4, 4))

import threading
class train_thread(threading.Thread):
  def run(self):
      _yolo_tarin_()
thread1 = train_thread()

# 根据本次的行动及其反馈（下一个时间步的状态），返回下一次的最佳行动
# epsilon_coefficient为贪心策略中的ε，取值范围[0,1]，取值越大，行为越随机
# 当epsilon_coefficient取值为0时，将完全按照q_table行动。故可作为训练模型与运用模型的开关值
def get_action(state, action, observation, reward, episode, epsilon_coefficient=0.0):
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
    return next_action, next_state

timer = time.time()
yolo = YOLO()
f = open("2.txt", 'w', encoding='utf-8')
#----------------------------------------------------------------------------------------
perception_queue = Queue()

episodes=[]
rewards=[]
n=1
flag = False
for episode in range(num_episodes):
    env.reset()  # 初始化本场游戏的环境
    episode_reward = 0  # 初始化本场游戏的得分
    q_table_cache = q_table # 创建q_table还原点，如若训练次数超次，则不作本次训练记录。
    v_state = 3
    state = env.state
    action = np.argmax(q_table[state])
    virtual_word = np.zeros([4, 4])

    for t in range(max_number_of_steps):
        #a_time = time.time()
        image = env.render(mode='rgb_array')    # 更新并渲染游戏画面
        plt.imsave('./figu/1.jpg',image)
        img = './figu/1.jpg'
        image = Image.open(img)

        #贴图
        #img = image_config(img)

        predicted_classes,boxes = yolo.detect_image(image,crop = False, count=False)
        print(predicted_classes)
        print(boxes)      
        #c_time = time.time()
        #print("检测时间：")
        #print(c_time - b_time)
        #print(rois)
        #print(id)
        for index in range(0,4):
            x = int((boxes[index][0]-100)/100)
            y = int((boxes[index][1]-100)/100)
            if predicted_classes[index]=='robot':
                v_state = pos[x][y]
            if predicted_classes[index]=='obstacle':
                virtual_word[x][y] = -100
            if predicted_classes[index]=='destination':
                virtual_word[x][y] = 100
        #print(v_state)
        virtual_word[int(v_state/4)][int(v_state%4)] = -20
        #print(virtual_word)
        print(virtual_word)
        flag = 0
        a_score = np.zeros([4])
        while flag<=size:
            flag = flag + 1
            key = "%d_%d" % (v_state, action)
            if key not in transfer:
                a_score[action] = -100
                get_action(v_state, action, v_state, -100, 0.5)
                action = np.argmax(q_table[v_state])
                continue
            else:
                next_state = transfer[key]
                if virtual_word[int(next_state/4)][int(next_state%4)]<0:
                    a_score[action] = virtual_word[int(next_state/4)][int(next_state%4)]
                    get_action(v_state,action,next_state,virtual_word[int(next_state/4)][int(next_state%4)],0.5)
                    action = np.argmax(q_table[v_state])
                    continue
                else:
                    break
        if flag>size:
            action = np.argmax(a_score)
        # action = np.random.choice([0, 1, 2, 3])  # 随机决定小车运动的方向
        observation, reward, done, info = env.step(action)  # 进行活动,并获取本次行动的反馈结果
        #如果没有出界且reward相差很大
        if observation!=-1 and abs(virtual_word[int(observation/4)][int(observation%4)] - reward) > 50:
            print(observation,action)
            label = 1
            l_x = (int(observation%4)+1)*126+8
            l_y = (int((observation+1)/4))*126+49
            r_x = l_x+100
            r_y = l_y+100
            image.save('./data/data_process/render_img/%d.jpg'%n)
            #perception_queue.put([img,[l_x,l_y,r_x,r_y],label])
            print(l_x,l_y,r_x,r_y)
            if n%7==0:
                if not os.path.exists('./data/train_val_data/kongzhi_val.txt'):
                    os.makedirs('./data/train_val_data/kongzhi_val.txt')
                list_file = open('./data/train_val_data/kongzhi_val.txt','a')#a为追加写入，w为覆盖写入，r为只读
                list_file.write('./data/data_process/render_img/%d.jpg' % (n))
                list_file.write(' '+'%d,%d,%d,%d,%d\n'%(l_x,l_y,r_x,r_y,label))  
                n=n+1
            else:
                if not os.path.exists('./data/train_val_data/kongzhi_train.txt'):
                    os.makedirs('./data/train_val_data/kongzhi_train.txt')
                list_file = open('./data/train_val_data/kongzhi_train.txt','a')#a为追加写入，w为覆盖写入，r为只读
                list_file.write('./data/data_process/render_img/%d.jpg' % (n))
                list_file.write(' '+'%d,%d,%d,%d,%d\n'%(l_x,l_y,r_x,r_y,label))  
                n=n+1
                flag = True

        #如果文件内容有新增：
        if flag == True:
            #新开一个线程？
            thread1.start()
            flag = False

        #如果文件夹中存在新的权重:
        if  os.path.exists('./logs/shibie/ep010.pth'):
            yolo = YOLO(model_path="./logs/shibie/ep010.pth")

        action, state = get_action(state, action, observation, reward, episode, 0.5)  # 作出下一次行动的决策
        episode_reward += reward
        if done:
            np.savetxt("q_table.txt", q_table, delimiter=",")
            print('已完成 %d 次训练，本次训练共进行 %d 步数。episode_reward：%d，平均分： %f' % (episode, t + 1, reward, last_time_steps.mean()))
            last_time_steps = np.hstack((last_time_steps[1:], [reward]))    # 更新最近100场游戏的得分stack
            break
        #d_time = time.time()
        #print("推理时间：")
        #print(d_time - c_time)

    if t == max_number_of_steps-1:
        q_table = q_table_cache # 超次还原q_table
        episode_reward = -100
        print('已完成 %d 次训练，本次训练共进行 %d 步数。episode_reward：%d，平均分： %f' % (episode, t + 1, reward, last_time_steps.mean()))
        last_time_steps = np.hstack((last_time_steps[1:], [reward]))  # 更新最近100场游戏的得分stack
    text = str(last_time_steps.mean())+'\n'
    f.write(text)

    if (last_time_steps.mean() >= goal_average_steps):
        np.savetxt("q_table.txt", q_table, delimiter=",")
        print('用时 %d s,训练 %d 次后，模型到达测试标准!' % (time.time() - timer, episode))
        env.close()
        sys.exit()

env.close()
sys.exit()