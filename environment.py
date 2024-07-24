import socket
import os
import gym
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import time
import sys
import json

def environment():
    sock_1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 绑定端口
    sock_1.bind(('127.0.0.1', 11661))  # env
    sensor_addr = ('127.0.0.1', 11662)  # sensor
    controller_addr = ('127.0.0.1', 11663)  # controller

    begin_flag = b'Framebegin:'  # 一张图像起始标志
    end_flag = b'Frameovers:'  # 一张图片结束标志
    package_len = 1024  # 包的大小

    num_episodes = 1500  # 共进行5000场游戏
    # 以栈的方式记录成绩
    max_number_of_steps = 250  # 每场游戏最大步数
    goal_average_steps = 100  # 平均分 达到平均分才算胜利
    num_consecutive_iterations = 100  # 栈的容量
    last_time_steps = np.zeros(num_consecutive_iterations)  # 只存储最近100场的得分（可以理解为是一个容量为100的栈）
    timer = time.time()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    root_dir = 'E:/2023-2024fall/yolov5-fuse-2.0/'
    env = gym.make('GridWorld-v1')

    for episode in range(num_episodes):
        env.reset()  # 初始化本场游戏的环境
        episode_reward = 0  # 初始化本场游戏的得分
        # state = env.state

        for t in range(max_number_of_steps):
            # a_time = time.time()
            image = env.render(mode='rgb_array')  # 更新并渲染游戏画面

            plt.imsave(root_dir + 'data/data_process/temporary_img/temporary.jpg', image)
            img = root_dir + 'data/data_process/temporary_img/temporary.jpg'
            image_ = Image.open(img)
            # image.show()
            # 发送图片
            np_img = cv2.cvtColor(np.asarray(image_), cv2.COLOR_RGB2BGR)
            # 将图片编码  send_data_num为0-255数字
            res, send_data_num = cv2.imencode('.jpg', np_img, (cv2.IMWRITE_JPEG_QUALITY, 50))
            send_data = send_data_num.tobytes()  # 编码转换为 字节流
            send_data_len = len(send_data)  # 获取数据长度
            i = send_data_len // package_len  # 分包发送 包的个数
            j = send_data_len % package_len  # 剩余字节的个数
            sock_1.sendto(begin_flag, sensor_addr)  # 发送起始标志
            for n in range(0, i):  # 分包发送
                sock_1.sendto(send_data[n * package_len:package_len * (n + 1)], sensor_addr)
            sock_1.sendto(send_data[-j:], sensor_addr)  # 发送剩余字节
            sock_1.sendto(end_flag, sensor_addr)  # 发送结束标志





            data, addr = sock_1.recvfrom(2048)
            # print(data)
            data = data.decode('utf-8')
            # print(data)
            # print(type(data))
            data = json.loads(data)
            # print(data)
            action = data['action']

            # print(boxes)
            # print(predicted_classes)
            observation, reward, done, info = env.step(action)

            # data = (observation,reward,done)
            data = {'observation':observation, 'reward':reward, 'done':done, 'episode':episode, 't':t}
            data = json.dumps(data)
            print(data)
            sock_1.sendto(bytes(data, encoding='utf-8'), controller_addr)


            episode_reward += reward
            if done:
                # np.savetxt("q_table.txt", q_table, delimiter=",")
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
            # q_table = q_table_cache  # 超次还原q_table
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
            # np.savetxt("q_table.txt", q_table, delimiter=",")
            print('用时 %d s,训练 %d 次后，模型到达测试标准!' % (time.time() - timer, episode))
            env.close()
            sys.exit()


if __name__ == '__main__':
    environment()
