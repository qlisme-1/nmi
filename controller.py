import socket
import numpy as np
import os
import json
import math



def controller():
    sock_3 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 绑定端口
    sock_3.bind(('127.0.0.1', 11663))  # controller
    sensor_addr = ('127.0.0.1', 11662)  # sensor
    env_addr = ('127.0.0.1', 11661)  # env
    root_dir = 'E:/2023-2024fall/yolov5-fuse-2.0/'

    size = 10
    q_table = np.random.uniform(low=-1, high=1, size=(size * size, 4))
    episode = 1
    q_table_cache = q_table

    def get_action(state, action, observation, reward, episode, epsilon_coefficient=0.2):
        # print(observation)
        next_state = observation
        epsilon = epsilon_coefficient * (0.99 ** episode)
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

    pos = []
    for i in range(size):
        post = []
        for j in range(size):
            post.append(j + i * size)
        pos.append(post)
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

    n = 1
    state = 0
    v_state = 3
    while True:
        # q_table_cache = q_table  # 创建q_table还原点，如若训练次数超次，则不作本次训练记录。
        # action = np.argmax(q_table[state])


        action = np.argmax(q_table[state])
        virtual_word = np.zeros([size, size])

        # 接收位置信息
        #################################
        data, addr = sock_3.recvfrom(4096)
        # print(data)
        data = data.decode('utf-8')
        # print(data)
        # print(type(data))
        data = json.loads(data)
        # print(data)
        boxes = data['boxes']
        predicted_classes = data['predicted_classes']
        # print(boxes)
        # print(predicted_classes)
        #################################

        for index in range(len(boxes)):
            x = int((boxes[index][0] - 50) / 50)
            y = int((boxes[index][1] - 50) / 50)
            # print(x,y)
            if predicted_classes[index] == 'robot':
                v_state = pos[x][y]  # 智能体的位置编号
            if predicted_classes[index] == 'obstacle':
                virtual_word[x][y] = -100  # 障碍物的位置信息设置奖励为-100
            if predicted_classes[index] == 'destination':  # 终点的位置信息并设置奖励为100
                virtual_word[x][y] = 100
        virtual_word[9][9] = 100

        # virtual_word[int(v_state / 10)][int(v_state % 10)] = -math.sqrt(
        #     (9 - int(v_state / 10)) ** 2 + (9 - int(v_state % 10)) ** 2)

        # print(virtual_word)

        flag = 0
        a_score = np.zeros([4])
        while flag < 4:
            flag = flag + 1
            key = "%d_%d" % (v_state, action)
            if key not in transfer:
                a_score[action] = -100
                get_action(v_state, action, v_state, -100, 0.5)
                # action = np.argmax(q_table[v_state])
                action = (action + 1) % 4
                continue
            else:
                next_state = transfer[key]
                if virtual_word[int(next_state / 10)][int(next_state % 10)] == -100:
                    a_score[action] = -100
                    get_action(v_state, action, v_state, -100, 0.5)
                    # action = np.argmax(q_table[v_state])
                    action = (action + 1) % 4
                elif virtual_word[int(next_state / 10)][int(next_state % 10)] < 0:
                    a_score[action] = virtual_word[int(next_state / 10)][int(next_state % 10)]
                    action = (action + 1) % 4
                    get_action(v_state, action, next_state,
                               virtual_word[int(next_state / 10)][int(next_state % 10)], 0.5)
                else:
                    for index in range(4):
                        if a_score[index] == 0:
                            a_score[index] = q_table[v_state][index]
                    # action = (action+1)%4
                    break
        if flag >= 4:
            epsilon_coefficient = 0.2  # 调大
            epsilon = epsilon_coefficient * (0.99 ** episode)  # 调小一点
            if epsilon <= np.random.uniform(0, 1):
                action = np.argmax(a_score)
            else:
                choindex = [i for i, e in enumerate(a_score) if e != -100]
                action = np.random.choice(choindex)

        # 传递action
        ###############################################
        data = {'action': int(action)}
        data = json.dumps(data)
        print(data)
        sock_3.sendto(bytes(data, encoding='utf-8'), env_addr)
        ###############################################

        # 接收真实奖励
        ###############################################
        data, addr = sock_3.recvfrom(2048)
        # print(data)
        data = data.decode('utf-8')
        # print(data)
        # print(type(data))
        data = json.loads(data)
        # print(data)
        observation = data['observation']
        reward = data['reward']
        done = data['done']
        episode = data['episode']
        t = data['t']
        # print(boxes)
        # print(predicted_classes)
        ###############################################

        if observation != -1 and abs(virtual_word[int(observation / size)][int(observation % size)] - reward) > 50:
            print(observation, action)
            label = 1

            l_x = (int(observation / 10)) * 50 + 50
            l_y = (int(observation % 10)) * 50 + 50
            r_x = l_x + 50
            r_y = l_y + 50
            # image_.save(root_dir + 'data/data_process/render_img/%d.jpg' % n)  # 上面写了
            # perception_queue.put([img,[l_x,l_y,r_x,r_y],label])
            # print(l_x,l_y,r_x,r_y)
            if n % 7 == 0:  # 分配训练集和验证集
                # 验证集
                if not os.path.exists(root_dir + 'data/data_process/train_val_data/kongzhi_val.txt'):
                    os.makedirs(root_dir + 'data/data_process/train_val_data/kongzhi_val.txt')
                list_file = open(root_dir + 'data/data_process/train_val_data/kongzhi_val.txt',
                                 'a')  # a为追加写入，w为覆盖写入，r为只读
                list_file.write(root_dir + 'data/data_process/render_img/%d.jpg' % (n))
                list_file.write(' ' + '%d,%d,%d,%d,%d\n' % (l_y, l_x, r_y, r_x, label))
                n = n + 1
            else:
                # 训练集
                if not os.path.exists(root_dir + 'data/data_process/train_val_data/kongzhi_train.txt'):
                    os.makedirs(root_dir + 'data/data_process/train_val_data/kongzhi_train.txt')
                list_file = open(root_dir + 'data/data_process/train_val_data/kongzhi_train.txt',
                                 'a')  # a为追加写入，w为覆盖写入，r为只读
                list_file.write(root_dir + 'data/data_process/render_img/%d.jpg' % (n))
                list_file.write(' ' + '%d,%d,%d,%d,%d\n' % (l_y, l_x, r_y, r_x, label))
                n = n + 1

        action, state = get_action(state, action, observation, reward, episode, 0.5)  # 作出下一次行动的决策
        # if done:
        #     q_table_cache = q_table
        #     state = 0
        #     print('done')
        # if t == 249:
        #     state = 0
        #     print('t==249')
        #     q_table = q_table_cache







if __name__ == '__main__':
    controller()
