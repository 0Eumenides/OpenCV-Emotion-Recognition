# Clone下来后运行`main.py`即可出结果


# 环境
1. [Python3安装](https://www.python.org/downloads/)
2. OpenCV库安装
在cmd中输入`pip install opencv-contrib-python`，如果显示pip不是内部或外部命令，那就是未配置环境变量，自行百度。
3. Numpy库安装
在cmd中输入`pip install numpy`
4. OpenCV人脸识别器下载(文件夹中已经有了，lbpcascade_frontalface.xml)
这里我选择的是LBPH人脸识别器，他能很好的避免光线明暗的影响
# 准备训练数据
这里我们在网上搜集了20多张有关小孩情绪的照片（精力有限，只作简单的学习），将其手动标记为`happy`和`sad`两种标签，照片命名规则为[标签][number].jpg。![在这里插入图片描述](https://img-blog.csdnimg.cn/20200311125044852.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1J1YW5lcw==,size_16,color_FFFFFF,t_70)
将照片分为训练数据和测试数据，分别放在`img_train`和`img_predict`文件夹下。![在这里插入图片描述](https://img-blog.csdnimg.cn/20200311125348514.png)
所以我们的目录结构是这样的

-- img_train<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- happy1.jpg<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- happy2.jpg<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- ...<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- sad1.jpg<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- sad2.jpg<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- ...<br>
-- img_predict<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- happy1.jpg<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- sad1.jpg<br>

准备工作已经做好了，接下来我们可以开始编写代码了，这里我将代码分成了6个源文件
**源文件**`detect_face.py`
```py
import cv2


def detect_face(img):
    #将图像转变成灰度图像，因为OpenCV人脸检测器需要灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)

    #加载OpenCV人脸识别器，注意这里的路径是前面下载识别器时，你保存的位置
    face_cascade = cv2.CascadeClassifier(r'D:\matlab\matlab\toolbox\vision\visionutilities\classifierdata\cascade\lbp\lbpcascade_frontalface.xml')

    #scaleFactor表示每次图像尺寸减小的比例，minNeighbors表示构成检测目标的相邻矩形的最小个数
    #这里选择图像尺寸减小1.2倍。minNeighbors越大，识别出来的人脸越准确，但也极易漏判
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=1)

    # 如果图中没有人脸，该图片不参与训练，返回None
    if len(faces) == 0:
        return None, None

    # 提取面部区域
    (x, y, w, h) = faces[0]

    #返回人脸及其所在区域
    return gray[y:y + w, x:x + h], faces[0]

```
*detect_face.py*文件中定义了函数`detect_face`，它能提取输入图像的人脸及其位置。

**源文件**`prepare_training_data.py`

```python
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

```
目前为止，我们对图片的预处理已经做好了！

# 识别器的训练
函数`cv2.face.LBPHFaceRecognizer_create()`将返回一个局部二值模式直方图（LBPH）人脸识别器
我们可以开始写**源文件**`main.py`了

```python
from prepare_training_data import prepare_training_data
import cv2
import numpy as np
#该文件我们稍后编写
from predict import predict

if __name__ == '__main__':
    print("Preparing data...")
    #调用之前写的函数，得到包含多个人脸矩阵的序列和它们对于的标签
    faces, labels = prepare_training_data()
    print("Data prepared")

    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))

    #得到（LBPH）人脸识别器
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    #应用数据，进行训练
    face_recognizer.train(faces, np.array(labels))
    print("Predicting images...")

    # 加载预测图像，这里我图简单，就直接把路径写上去了
    test_img1 = cv2.imread(r"./img_predict/happy1.jpg",0)
    test_img2 = cv2.imread(r"./img_predict/sad1.jpg",0)

    # 进行预测
    # 注意，该函数我们还未编写！！！
    predicted_img1 = predict(test_img1, face_recognizer)
    predicted_img2 = predict(test_img2, face_recognizer)
    print("Prediction complete")

    # 显示预测结果
    cv2.imshow('Happy', predicted_img1)
    cv2.imshow('Sad', predicted_img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

```

相信到目前为止你已经对opencv的识别过程有了个清晰的认识，接下来我们将编写`predict`函数。
# 预测
**源文件**`predict.py`
```python
from detect_face import detect_face

# 这两个文件我们还未编写
from draw_rectangle import draw_rectangle
from draw_text import draw_text


def predict(test_img, face_recognizer):
    # 将标签1，2转换成文字
    subjects = ['', 'Happy', 'Sad']

    # 得到图像副本
    img = test_img.copy()

    # 从图像中检测脸部
    face, rect = detect_face(img)

    # 使用我们的脸部识别器预测图像
    label = face_recognizer.predict(face)
    # 获取由人脸识别器返回的相应标签的名称
    label_text = subjects[label[0]]

    # 注意，下面两个函数我们还未编写！！！
    # 在检测到的脸部周围画一个矩形
    draw_rectangle(img, rect)
    # 在矩形周围标出人脸情绪
    draw_text(img, label_text, rect[0], rect[1] - 5)

    return img

```
程序的主要部分我们已经完成，还差`draw_rectangle`，和`draw_text`两个函数，我们将其完成：

**源文件**`draw_rectangle.py`

```python
import cv2

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
```

**源文件**`draw_text.py`

```python
import cv2

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
```

大功告成！接下来我们运行`main.py`，得到识别结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200311181026991.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1J1YW5lcw==,size_16,color_FFFFFF,t_70)

可以看到识别结果还不错，如果增加训练集应该能训练出一个不错的情绪识别器。

# Reference
1. [https://www.cnblogs.com/zhuifeng-mayi/p/9171383.html?tdsourcetag=s_pctim_aiomsg](https://www.cnblogs.com/zhuifeng-mayi/p/9171383.html?tdsourcetag=s_pctim_aiomsg)
2. [https://blog.csdn.net/firstlai/article/details/70882240](https://blog.csdn.net/firstlai/article/details/70882240)
3.  [https://blog.csdn.net/Young__Fan/article/details/80022860](https://blog.csdn.net/Young__Fan/article/details/80022860)
4. [http://blog.sina.com.cn/s/blog_9fcb9cbb01012b5b.html](http://blog.sina.com.cn/s/blog_9fcb9cbb01012b5b.html)
5. [https://blog.csdn.net/weixin_42309501/article/details/80781293](https://blog.csdn.net/weixin_42309501/article/details/80781293)
