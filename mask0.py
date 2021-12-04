import cv2
import time
import RPi.GPIO
import threading

RPi.GPIO.setwarnings(False)

RPi.GPIO.setmode(RPi.GPIO.BCM)

RPi.GPIO.setup(26, RPi.GPIO.OUT)
RPi.GPIO.output(26, False)

RPi.GPIO.setup(4, RPi.GPIO.OUT)
p=RPi.GPIO.PWM(4, 50)


# 导入人脸模型
face_H = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# 导入训练好的戴口罩人脸模型
mask_C = cv2.CascadeClassifier('cascade.xml')


def motor():
    for i in range(1):
        p.start(0)
        p.ChangeDutyCycle(11.1)
        time.sleep(2)
        p.ChangeDutyCycle(6.1)
        time.sleep(2)
        p.ChangeDutyCycle(0)


# 帧率
def Fpsgain():
    global fps
    global counter
    global start_time
    counter += 1
    if (time.time() - start_time) > 0.1:
        fps = counter / (time.time() - start_time)
        counter = 0
        start_time = time.time()
    cv2.putText(img, 'FPS:' + str(int(fps)), (260, 235), font, 0.5, (0, 0, 255), 2)


# 相机
def Setcamera(cap):
    # 设置摄像头获取视频编码格式
    cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    # 设置窗口的大小
    cap.set(3, 320)
    cap.set(4, 240)


# 调用摄像头摄像头
cap = cv2.VideoCapture(0)
Setcamera(cap)

# 字体
font = cv2.FONT_HERSHEY_SIMPLEX
start_time = time.time()
fps = 0
counter = 0

while True:
    # 获取摄像头的画面
    # 会得到两个参数，一个是否捕捉到图像（True/False），另一个为存放每帧的图像
    # 添加一个线程，用于驱动舵机
    add_thread = threading.Thread(target=motor)
    ret, img = cap.read()  # 读取一帧图像
    # 每帧图像放大1.08倍，重复检测10次
    # 得到人脸，可能不止一个
    faces = face_H.detectMultiScale(img, 1.08, 10)
    mask = mask_C.detectMultiScale(img, 1.08, 10)
    for (x, y, w, h) in faces:
        # 画出人脸框,蓝色，画笔宽度为1
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 64, 64), 1)
        cv2.putText(img, 'no mask', (x, y - 5), font, 0.5, (255, 64, 64), 1)
        RPi.GPIO.output(26, True)

    for (ex, ey, ew, eh) in mask:
        # 画出口罩框，绿色，画笔宽度为1
        cv2.rectangle(img, (int(ex), int(ey)), (int(ex + ew), int(ey + eh)), (0, 255, 0), 1)
        cv2.putText(img, 'mask', (ex, ey - 5), font, 0.5, (0, 255, 0), 1)
        RPi.GPIO.output(26, False)
        
        # 识别到戴口罩，开启线程，转动舵机
        add_thread.start()
        cv2.waitKey(1000)
        # 关闭线程
        add_thread.join()

    # 计算帧率
    Fpsgain()
    # 实时展示画面
    cv2.imshow('img', img)
    # 每5毫秒检测一次键盘
    # 当按下"q"键时退出检测
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# 关闭所有窗口
cap.release()
# 释放资源
cv2.destroyAllWindows()
