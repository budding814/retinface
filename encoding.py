import os

from retinaface import Retinaface

'''
在更换facenet网络后一定要重新进行人脸编码，运行encoding.py。
'''
retinaface = Retinaface(1)

list_dir = os.listdir("lfw////")
image_paths = []
names = []
i=0
####
for name in list_dir:
    list_dir1=os.listdir("lfw////"+name)
    for name1 in list_dir1:
        #print(name1.split("_")[0]+"_"+name1.split("_")[1])
        #print("lfw//"+name+"//"+name1)
        #path="lfw/"+name+"/"+name1
        #image = np.array(Image.open(path), np.float32
        image_paths.append("lfw////"+name+"////"+name1)
        names.append(name)

        i=i+1
        if i==6407:
            print("lfw////"+name+"////"+name1)

####

#print(names[1])
retinaface.encode_face_dataset(image_paths,names)
