import sys, os

sys.path.append("../")
import gym
import numpy as np
import time
import sys
import PIL
from yolo import YOLO
import argparse
from PIL import Image
# from queue import Queue
# from queue import Empty
import matplotlib.pyplot as plt
import matplotlib as mpl
import os




def fuse(path, envname, param):
    # log_file = "../logs/" + str(envname)+'_'+str(param) + ".log"
    # log = Logger(log_file, level='debug')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    # --------------------------------------------------------------------------
    #                             根目录设置
    # --------------------------------------------------------------------------
    root_dir = path
    # --------------------------------------------------------------------------
    #                             游戏环境加载
    # --------------------------------------------------------------------------
    env = gym.make(envname)
    # --------------------------------------------------------------------------
    #                              游戏设置
    # --------------------------------------------------------------------------
    size = 10
    num_episodes = 1500  # 共进行5000场游戏
    # 以栈的方式记录成绩
    max_number_of_steps = 250  # 每场游戏最大步数
    goal_average_steps = 100  # 平均分 达到平均分才算胜利
    num_consecutive_iterations = 100  # 栈的容量
    last_time_steps = np.zeros(num_consecutive_iterations)  # 只存储最近100场的得分（可以理解为是一个容量为100的栈）
    q_table = np.random.uniform(low=-1, high=1, size=(size * size, 4))
    vfile = open(root_dir + 'vworld.txt', 'w').close()

    # --------------------------------------------------------------------------
    # 根据本次的行动及其反馈（下一个时间步的状态），返回下一次的最佳行动
    # epsilon_coefficient为贪心策略中的ε，取值范围[0,1]，取值越大，行为越随机
    # 当epsilon_coefficient取值为0时，将完全按照q_table行动。故可作为训练模型与运用模型的开关值。
    # --------------------------------------------------------------------------
    def get_action(state, action, observation, reward, episode, epsilon_coefficient=0.2):
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

    # pos = []
    # for i in range(size):
    #     for j in range(size):
    #         pos.append(j+i*10)
    # transfer = dict()

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

    timer = time.time()
    yolo = YOLO()
    # f = open("2.txt", 'w', encoding='utf-8')
    # ----------------------------------------------------------------------------------------
    # perception_queue = Queue()

    episodes = []
    rewards = []
    n = 1

    for episode in range(num_episodes):
        env.reset()  # 初始化本场游戏的环境
        episode_reward = 0  # 初始化本场游戏的得分
        q_table_cache = q_table  # 创建q_table还原点，如若训练次数超次，则不作本次训练记录。
        v_state = 3  # 虚拟世界状态
        state = env.state
        print("state",state)
        action = np.argmax(q_table[state])
        virtual_word = np.zeros([size, size])

        for t in range(max_number_of_steps):
            # a_time = time.time()
            image = env.render(mode='rgb_array')  # 更新并渲染游戏画面

            plt.imsave(root_dir + 'data/data_process/temporary_img/temporary.jpg', image)
            img = root_dir + 'data/data_process/temporary_img/temporary.jpg'
            image_ = Image.open(img)

            # image.show()
            image = image_.copy()
            print(type(image))
            predicted_classes, boxes = yolo.detect_image(image, crop=False, count=False)
            # image.show()

            # print(boxes)

            # print(predicted_classes)
            # print(boxes)
            # c_time = time.time()
            # print("检测时间：")
            # print(c_time - b_time)
            # print(rois)
            # print(id)

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
            # print(v_state)
            # if virtual_word[int(v_state/10)][int(v_state%10)]>-50:
            #     virtual_word[int(v_state/10)][int(v_state%10)] = -10+virtual_word[int(v_state/10)][int(v_state%10)]
            # #print(virtual_word)
            # print(virtual_word)
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
                epsilon_coefficient = 0.2
                epsilon = epsilon_coefficient * (0.99 ** episode)  # ε-贪心策略中的ε
                if epsilon <= np.random.uniform(0, 1):
                    action = np.argmax(a_score)
                else:
                    choindex = [i for i, e in enumerate(a_score) if e != -100]
                    action = np.random.choice(choindex)
                # action = 3
            # vfile = open(root_dir+'vworld.txt','a')#a为追加写入，w为覆盖写入，r为只读
            # vfile.write(str(a_score))
            # vfile.write(str(action)+'\n')
            # vfile.close()
            # action = np.random.choice([0, 1, 2, 3])  # 随机决定小车运动的方向
            # print(virtual_word)
            observation, reward, done, info = env.step(action)  # 进行活动,并获取本次行动的反馈结果
            # 如果没有出界且reward相差很大
            # observation是在真实环境中执行动作之后得到的下一个状态
            # 指的是虚拟世界的对应位置的reward和真实世界得到的reward差距很大
            # 说明没有感知出来或者感知出错需要增加新的标签到训练集和验证集中去
            # label=1表示的是obstacle

            if observation != -1 and abs(virtual_word[int(observation / size)][int(observation % size)] - reward) > 50:
                print(observation, action)
                label = 1

                l_x = (int(observation / 10)) * 50 + 50
                l_y = (int(observation % 10)) * 50 + 50
                r_x = l_x + 50
                r_y = l_y + 50
                image_.save(root_dir + 'data/data_process/render_img/%d.jpg' % n)
                # perception_queue.put([img,[l_x,l_y,r_x,r_y],label])
                # print(l_x,l_y,r_x,r_y)
                # 分配训练集和验证集
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
            episode_reward += reward
            if done:
                np.savetxt("q_table.txt", q_table, delimiter=",")
                print('已完成 %d 次训练，本次训练共进行 %d 步数。episode_reward：%d，平均分： %f' % (
                episode, t + 1, reward, last_time_steps.mean()))
                # ------------------------------------------------
                #         绘制平均分关于迭代次数的图像
                # ------------------------------------------------
                draw_file_1 = open(root_dir + 'data/draw/1010/fuse_single_episode.txt', 'a')
                draw_file_2 = open(root_dir + 'data/draw/1010/fuse_single_reward.txt', 'a')
                draw_file_1.write('%d\n' % episode)
                draw_file_2.write('%d\n' % last_time_steps.mean())

                last_time_steps = np.hstack((last_time_steps[1:], [reward]))  # 更新最近100场游戏的得分stack
                break
            # d_time = time.time()
            # print("推理时间：")
            # print(d_time - c_time)

        if t == max_number_of_steps - 1:  # 步数超过给定的最大步数了
            q_table = q_table_cache  # 超次还原q_table
            episode_reward = -100
            print(
                '已完成 %d 次训练，本次训练共进行 %d 步数。episode_reward：%d，平均分： %f' % (episode, t + 1, reward, last_time_steps.mean()))
            last_time_steps = np.hstack((last_time_steps[1:], [reward]))  # 更新最近100场游戏的得分stack

            # ------------------------------------------------
            #         绘制平均分关于迭代次数的图像
            # ------------------------------------------------
            draw_file_1 = open(root_dir + 'data/draw/1010/fuse_single_episode.txt', 'a')
            draw_file_2 = open(root_dir + 'data/draw/1010/fuse_single_reward.txt', 'a')
            draw_file_1.write('%d\n' % episode)
            draw_file_2.write('%d\n' % last_time_steps.mean())

        if (last_time_steps.mean() >= goal_average_steps):
            np.savetxt("q_table.txt", q_table, delimiter=",")
            print('用时 %d s,训练 %d 次后，模型到达测试标准!' % (time.time() - timer, episode))
            env.close()
            sys.exit()
    env.close()
    sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--obu', type=str, default='4', help='specify the number of this vehicle demo,example:3')
    parser.add_argument('--env', type=str, default='GridWorld-v1', help='env name')

    parser.add_argument('--param', type=str, default='sig', help='mod')
    parser.add_argument('--path', type=str, default='E:/2023-2024fall/yolov5-fuse-2.0/', help='root path')
    args = parser.parse_args()
    draw_file_1 = open(args.path + 'data/draw/1010/fuse_single_episode.txt', 'a')
    draw_file_2 = open(args.path + 'data/draw/1010/fuse_single_reward.txt', 'a')
    draw_file_1.truncate(0)
    draw_file_2.truncate(0)
    fuse(args.path, args.env, args.param)