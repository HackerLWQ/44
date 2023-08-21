import dlib
import cv2
import threading
import queue

def start_tracker(box, label, rgb, inputQueue, outputQueue):
    t = dlib.correlation_tracker()
    rect = dlib.rectangle(int(box[0]), int(box[1]), int(box[0]+box[2]), int(box[1]+box[3]))
    t.start_track(rgb, rect)

    while True:
        rgb = inputQueue.get()

        if rgb is not None:
            t.update(rgb)
            pos = t.get_position()

            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            outputQueue.put((label, (startX, startY, endX, endY)))

# 配置参数
args = {
    "video": "videos/los_angeles.mp4"
}

# 加载视频流
vs = cv2.VideoCapture(args["video"])

# 创建输入和输出队列
inputQueue = queue.Queue(maxsize=10)
outputQueue = queue.Queue(maxsize=10)

# 手动选择初始边界框（多个目标的话，可以使用一个循环来选择多个框）
grabbed, frame = vs.read()
if not grabbed:
    exit()
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
boxes = []  # 存储选中的多个框
while True:
    box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
    if box == (0, 0, 0, 0):
        break  # 如果没有选择框，则退出选择模式
    boxes.append(box)
print("Selected boxes:", boxes)

# 启动目标跟踪线程
tracking_threads = []
for idx, box in enumerate(boxes):
    label = f"target_{idx}"
    tracking_thread = threading.Thread(target=start_tracker,
                                       args=(box, label, rgb_frame, inputQueue, outputQueue))
    tracking_thread.daemon = True
    tracking_thread.start()
    tracking_threads.append(tracking_thread)

while True:
    grabbed, frame = vs.read()
    if not grabbed:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputQueue.put(rgb_frame)

    while not outputQueue.empty():
        label, box = outputQueue.get()
        startX, startY, endX, endY = box
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

vs.release()
cv2.destroyAllWindows()
