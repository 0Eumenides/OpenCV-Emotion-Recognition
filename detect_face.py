import cv2


def detect_face(img):
    #将图像转变成灰度图像，因为OpenCV人脸检测器需要灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)

    #加载OpenCV人脸识别器，注意这里的路径是前面下载识别器时，你保存的位置
    face_cascade = cv2.CascadeClassifier(r'lbpcascade_frontalface.xml')

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
