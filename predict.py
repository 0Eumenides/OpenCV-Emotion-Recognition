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
