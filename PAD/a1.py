import os
import sys
from PIL import Image
import numpy as np
import cv2

def getImageAndLabels():
    #建两个空列表后续存储数据
    facesSamples=[]
    ids=[]
    imagePaths=[]
    list_dir = os.listdir("D:\\myProject\lfw")
    for name in list_dir:
        list_dir1 = os.listdir("D:\\myProject\lfw\\" + name)
        for name1 in list_dir1:
            imagePaths.append("D:\\myProject\lfw\\" + name+"\\"+name1)

    #检测人脸
    #print(1)
    face_detector = cv2.CascadeClassifier('D:\\myProject\\cascades\\haarcascade_frontalface_alt2.xml')
    #打印数组imagePaths
    print('路径：',imagePaths)
    id=0
    #遍历列表中的图片
    for imagePath in imagePaths:
        if id>100:
            break
        #打开图片,灰度
        PIL_img=Image.open(imagePath).convert('L')
        #print(PIL_img)
        #此时获取的是整张图片的数组
        img_numpy=np.array(PIL_img,'uint8')
        #print(img_numpy)
        #获取图片人脸特征，相当于rio
        faces = face_detector.detectMultiScale(img_numpy)
        #将文件名前的名字转化为ID并记录下来
        str_id = os.path.split(imagePath)[1].split('.')[0]
        #id = str_id
        len1 = len(str_id.split("_"))
        str1 = str_id.split("_")[0]

        for i in range(1, len1 - 1):
            str1 = str1 + "_" + str_id.split("_")[i]
        id=id+1
        #id = os.path.split(imagePath)[1].split('.')[0]
        #预防检测到无面容照片
        for x,y,w,h in faces:
            #把ID写进ids列表中
            ids.append(id)
            #把所画的方框写进facesSamples列表中
            facesSamples.append(img_numpy[y:y+h,x:x+w])
        #打印脸部特征和id
        print('id:', id)
    print('fs:', facesSamples)
    return facesSamples,ids

if __name__ == '__main__':
    list_dir = os.listdir("D:\myProject\lfw")
    #for name in list_dir:
    #list_dir1 = os.listdir("D:\myProject\lfw\\" + name)
    #path="D:\myProject\lfw\\"
    faces, ids = getImageAndLabels()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
        # 保存文件
    recognizer.write('D:\myProject\lfw_trainer\\trainer_x.yml')
