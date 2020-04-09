import os

#引入刚刚编写好的源文件
from detect_face import detect_face
import cv2


def prepare_training_data():
    #读取训练文件夹中的图片名称
    dirs = os.listdir(r'./img_train')
    faces = []
    labels = []
    for image_path in dirs:
        #如果图片的名称以happy开头，则标签为1l；sad开头，标签为2
        if image_path[0] == 'h':
            label = 1
        else:
            label = 2

        #得到图片路径
        image_path = './img_train/' + image_path

        #返回灰度图，返回Mat对象
        image = cv2.imread(image_path,0)

        #以窗口形式显示图像，显示100毫秒
        cv2.imshow("Training on image...", image)
        cv2.waitKey(100)

        #调用我们先前写的函数
        face, rect = detect_face(image)
        if face is not None:
            faces.append(face)
            labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels
