import cv2
import time
import RPi.GPIO
import threading

RPi.GPIO.setwarnings(False)

RPi.GPIO.setmode(RPi.GPIO.BCM)

# LED灯
RPi.GPIO.setup(26, RPi.GPIO.OUT)
RPi.GPIO.output(26, False)

# 蜂鸣器
RPi.GPIO.setup(19, RPi.GPIO.OUT)
RPi.GPIO.output(19, RPi.GPIO.LOW)

# 舵机
RPi.GPIO.setup(16, RPi.GPIO.OUT)
p = RPi.GPIO.PWM(16, 50)

# 导入人脸模型
face_H = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# 导入训练好的戴口罩人脸模型
mask_C = cv2.CascadeClassifier('cascade.xml')


# 舵机
def duoji():
    for i in range(1):
        p.start(0)
        p.ChangeDutyCycle(11.1)
        time.sleep(5)
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
    cv2.putText(image, 'FPS:' + str(int(fps)), (260, 235), font, 0.5, (0, 0, 255), 2)


# 相机
class Camera:

    def __init__(self, camera):
        self.frame = []
        self.ret = False
        self.cap = object
        self.camera = camera
        self.openflag = False

    def open(self):
        self.cap = cv2.VideoCapture(self.camera)
        self.ret = self.cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.ret = self.cap.set(3, 320)
        self.ret = self.cap.set(4, 240)
        self.ret = False
        self.openflag = True
        threading.Thread(target=self.queryframe, args=()).start()

    def queryframe(self):
        while self.openflag:
            self.ret, self.frame = self.cap.read()

    def getframe(self):
        return self.ret, self.frame

    def close(self):
        self.openflag = False
        self.cap.release()


# 调用摄像头
camera = Camera(0)
camera.open()

# 字体
font = cv2.FONT_HERSHEY_SIMPLEX
start_time = time.time()
fps = 0
counter = 0

while True:
    # 添加一个线程，用于驱动舵机
    add_thread = threading.Thread(target=duoji)
    # 获取摄像头的画面
    # 会得到两个参数，一个是否捕捉到图像（True/False），另一个为存放每帧的图像
    # 读取一帧图像
    ret, image = camera.getframe()
    # 每帧图像放大1.1倍，重复检测10次
    # 得到人脸，可能不止一个
    if ret:
        faces = face_H.detectMultiScale(image, 1.1, 10)
        mask = mask_C.detectMultiScale(image, 1.1, 10)
        for (x, y, w, h) in faces:
            # 画出人脸框,蓝色，画笔宽度为1
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 64, 64), 1)
            cv2.putText(image, 'no mask', (x, y - 5), font, 0.5, (255, 64, 64), 1)
            # LED点亮
            RPi.GPIO.output(26, True)
            # 蜂鸣器响起
            RPi.GPIO.output(19, RPi.GPIO.HIGH)

        for (ex, ey, ew, eh) in mask:
            # 画出口罩框，绿色，画笔宽度为1
            cv2.rectangle(image, (int(ex), int(ey)), (int(ex + ew), int(ey + eh)), (0, 255, 0), 1)
            cv2.putText(image, 'mask', (ex, ey - 5), font, 0.5, (0, 255, 0), 1)
            # 识别到戴口罩，LED不亮，蜂鸣器不响，开启线程，转动舵机
            cv2.imshow('img', image)
            cv2.waitKey(100)
            
            # LED熄灭
            RPi.GPIO.output(26, False)
            # 蜂鸣器静音
            RPi.GPIO.output(19, RPi.GPIO.LOW)
            # 开启线程，转动舵机
            add_thread.start()
            # 关闭线程
            add_thread.join()

        # 计算帧率
        Fpsgain()
        # 实时展示画面
        cv2.imshow('img', image)
        # 每5毫秒检测一次键盘
        # 当按下"q"键时退出检测
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# 关闭所有窗口
camera.close()
# 释放资源
cv2.destroyAllWindows()
