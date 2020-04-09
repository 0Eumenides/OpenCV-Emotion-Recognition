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
