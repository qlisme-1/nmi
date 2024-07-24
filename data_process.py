from PIL import Image
import numpy as np
import cv2
# a = 100
# b = 100
# mother_image= Image.open("./data/big.png")  #大图地址
# im_crop=Image.open("./data/small.png") #子图地址
# mother_image.paste(im_crop,(a,b))  #括号中为左上角坐标,a为x坐标，b为y坐标
# mother_image.save("./data/merge.png")  #图片保存路径

def image_paste(mother_image,im_crop,x,y):
    mother_image = Image.fromarray(mother_image)
    im_crop = Image.fromarray(im_crop)
    mother_image.paste(im_crop, (x,y))  # 括号中为左上角坐标,a为x坐标，b为y坐标
    arr = np.array(mother_image)
    mother_image.save("./data/merge.png")  # 图片保存路径
    return arr

def image_config(mother_image):
    #mother_image = Image.fromarray(mother_image)
    #robotrans = Image.open("./image/robotrans.png") #子图地址
    fire1 = Image.open("./image/car1.png")
    fire2 = Image.open("./image/car2.png")
    unknow = Image.open("./image/unknow.png")
    diamond = Image.open("./image/diamond.png")
    mother_image.paste(fire1, (400, 300))  # 括号中为左上角坐标,a为x坐标，b为y坐标
    mother_image.paste(fire2, (100, 400))  # 括号中为左上角坐标,a为x坐标，b为y坐标
    mother_image.paste(unknow, (400, 200))  # 括号中为左上角坐标,a为x坐标，b为y坐标
    mother_image.paste(diamond, (400, 400))  # 括号中为左上角坐标,a为x坐标，b为y坐标
    mother_image.save("./image/merge.png")  # 图片保存路径
    # arr = np.array(mother_image)
    # return arr
    return mother_image

#图片旋转
def opencv_rotate(img, angle):
    """
    图片旋转，默认应该是逆时针转动
    :param img:
    :param angle:
    :return:
    """
    h, w = img.shape[:2]  # 图像的（行数，列数，色彩通道数）
    borderValue = (0, 0, 0, 0)
    # 颜色空间转换?
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    elif img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    center = (w / 2, h / 2)
    scale = 1.0
    # 2.1获取M矩阵
    """
    M矩阵
    [
    cosA -sinA (1-cosA)*centerX+sinA*centerY
    sinA cosA  -sinA*centerX+(1-cosA)*centerY
    ]
    """
    # cv2.getRotationMatrix2D(获得仿射变化矩阵)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    # 2.2 新的宽高，radians(angle) 把角度转为弧度 sin(弧度)
    new_H = int(
        w * np.fabs(np.sin(np.radians(angle))) + h * np.fabs(np.cos(np.radians(angle)))
    )
    new_W = int(
        h * np.fabs(np.sin(np.radians(angle))) + w * np.fabs(np.cos(np.radians(angle)))
    )
    # 2.3 平移
    M[0, 2] += (new_W - w) / 2
    M[1, 2] += (new_H - h) / 2

    # cv2.warpAffine(进行仿射变化)
    rotate = cv2.warpAffine(img, M, (new_W, new_H), borderValue=borderValue)
    return rotate

