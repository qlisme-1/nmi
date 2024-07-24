import json
import socket
from yolo import YOLO
import numpy as np
import cv2
from PIL import Image
# import json

def sensor():
    yolo = YOLO()

    sock_2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 绑定端口
    sock_2.bind(('127.0.0.1', 11662))  # env
    env_addr = ('127.0.0.1', 11661)  # env
    controller_addr = ('127.0.0.1', 11663)  # controller

    begin_flag = b'Framebegin:'  # 一张图像起始标志
    end_flag = b'Frameovers:'  # 一张图片结束标志
    package_len = 1024  # 包的大小
    img_bytes = b''  # 保存一张图像

    while True:
        data, addr = sock_2.recvfrom(package_len)
        if data:  # 判断data不为空
            if data == begin_flag:  # 如果图像数据包来了
                while True:  # 开启一个循环接受数据
                    data, addr = sock_2.recvfrom(package_len)
                    # 如果结束包来了  则说明一张图像数据完毕  udp 结束包是单独过来的 tcp 结束包会和最后一个图像数据包混在一起
                    if data == end_flag:
                        break  # 跳出接收这样图像的循环
                    img_bytes = img_bytes + data  # 如果不是结束包 则将数据添加到变量 继续循环
                # 显示图片
                np_data = np.frombuffer(img_bytes, dtype="uint8")
                r_img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
                # r_img = r_img.reshape(pic_height, pic_width, 3)  # 会报错
                # cv2.imshow("title", r_img)
                # cv2.waitKey()
                # print(type(r_img))
                r_img = cv2.cvtColor(np.asarray(r_img), cv2.COLOR_RGB2BGR)
                image = Image.fromarray(r_img)
                # image.show()
                img_bytes = b''
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                predicted_classes, boxes = yolo.detect_image(image, crop=False, count=False)
                # image.show()
                # print(predicted_classes)
                # print(boxes)
                for i in range(0,len(boxes)):
                    boxes[i] = boxes[i].tolist()
                data = {'predicted_classes':predicted_classes, 'boxes':boxes}

                data = json.dumps(data)
                print(data)
                sock_2.sendto(bytes(data,encoding='utf-8'), controller_addr)
            else:
                print(data)
    # cv2.destroyAllWindows()



if __name__ == '__main__':
    sensor()