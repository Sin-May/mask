import cv2
import time

face_engine = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # 导入人脸级联分类器，'.xml'文件里包含训练出来的人脸特征
mask_cascade = cv2.CascadeClassifier('cascade.xml') # 导入人眼级联分类器，'.xml'文件里包含训练出来的人眼特征

#帧率计算
def Fpsgain():
    global fps
    global counter
    global start_time
    counter += 1
    if((time.time()-start_time)) > 0.1:
        fps = counter / (time.time() - start_time)
        counter = 0
        start_time = time.time()
    cv2.putText(img, 'FPS:'+str(int(fps)), (260,235), font, 0.5, (0, 0, 255), 2)

#相机参数设置
def Setcamera(cap):
    cap.set(6,cv2.VideoWriter.fourcc('M','J','P','G'))  #设置摄像头获取视频编码格式
    cap.set(3,320)
    cap.set(4,240)  #设置窗口的大小

cap = cv2.VideoCapture(0)   # 调用摄像头摄像头
Setcamera(cap)

font = cv2.FONT_HERSHEY_SIMPLEX  #字体
start_time = time.time()
fps = 0
counter = 0

while(True):
    # 获取摄像头拍摄到的画面
    # 会得到两个参数，一个是否捕捉到图像（True/False），另一个为存放每帧的图像
    ret, img= cap.read()  # 读取一帧图像
    # 每帧图像放大1.1倍，重复检测10次
    faces = face_engine.detectMultiScale(img,1.1,10)  # 得到人脸，可能不止一个
    mask = mask_cascade.detectMultiScale(img,1.1,8)
    for (x,y,w,h) in faces:
        # 画出人脸框,蓝色，画笔宽度为2
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,64,64),1)
        cv2.putText(img, 'no mask', (x, y - 5), font, 0.5, (255,64,64), 1)
        #print('face_x=',x,'face_y=',y)
    for (ex,ey,ew,eh) in mask:
        #画出口罩框，绿色，画笔宽度为1
        cv2.rectangle(img,(int(ex),int(ey)),(int(ex+ew),int(ey+eh)),(0,255,0),1)
        cv2.putText(img, 'mask', (ex, ey - 5), font, 0.5, (0,255,0), 1)
    #计算帧率
    Fpsgain()
    # 实时展示效果画面
    cv2.imshow('img',img)
    # 每5毫秒监听一次键盘动作
    if cv2.waitKey(5) & 0xFF == ord('q'):  #当按下“q”键时退出人脸检测
        break

# 最后，关闭所有窗口
cap.release()
cv2.destroyAllWindows() # 释放资源

