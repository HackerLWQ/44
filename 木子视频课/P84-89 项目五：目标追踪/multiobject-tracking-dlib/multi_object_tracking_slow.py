#导入工具包
from utils import FPS
import numpy as np
import argparse
import dlib
import cv2
"""
--prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt 
--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel 
--video race.mp4
"""
# 参数
prototxt_path = "mobilenet_ssd/MobileNetSSD_deploy.prototxt"
model_path = "mobilenet_ssd/MobileNetSSD_deploy.caffemodel"
video_path = "race.mp4"
output_path = "output.mp4"  # 可选的输出视频文件路径
confidence_threshold = 0.2

# 加载模型
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# 打开视频
vs = cv2.VideoCapture(video_path)


# SSD标签
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# 读取网络模型
print("[INFO] loading model...")

writer = None

# 一会要追踪多个目标
trackers = []
labels = []

# 计算FPS
fps = FPS().start()

while True:
	# 读取一帧
	(grabbed, frame) = vs.read()

	# 是否是最后了
	if frame is None:
		break

	# 预处理操作
	(h, w) = frame.shape[:2]
	width=600
	r = width / float(w)
	dim = (width, int(h * r))
	frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# 如果要将结果保存的话
	if output_path is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 或者尝试 None


		writer = cv2.VideoWriter(output_path, fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

	# 先检测 再追踪
	if len(trackers) == 0:
		# 获取blob数据
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)

		# 得到检测结果
		net.setInput(blob)
		detections = net.forward()

		# 遍历得到的检测结果
		for i in np.arange(0, detections.shape[2]):
			# 能检测到多个结果，只保留概率高的
			confidence = detections[0, 0, i, 2]

			# 过滤
			if confidence > confidence_threshold:
				# extract the index of the class label from the
				# detections list
				idx = int(detections[0, 0, i, 1])
				label = CLASSES[idx]

				# 只保留人的
				if CLASSES[idx] != "person":
					continue

				# 得到BBOX
				#print (detections[0, 0, i, 3:7])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# 使用dlib来进行目标追踪
				#http://dlib.net/python/index.html#dlib.correlation_tracker
				t = dlib.correlation_tracker()
				rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
				t.start_track(rgb, rect)

				# 保存结果
				labels.append(label)
				trackers.append(t)

				# 绘图
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 255, 0), 2)
				cv2.putText(frame, label, (startX, startY - 15),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

	# 如果已经有了框，就可以直接追踪了
	else:
		# 每一个追踪器都要进行更新
		for (t, l) in zip(trackers, labels):
			t.update(rgb)
			pos = t.get_position()

			# 得到位置
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			# 画出来
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 255, 0), 2)
			cv2.putText(frame, l, (startX, startY - 15),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

	# 也可以把结果保存下来
	if writer is not None:
		writer.write(frame)

	# 显示
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# 退出
	if key == 27:
		break

	# 计算FPS
	fps.update()



fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

if writer is not None:
	writer.release()

cv2.destroyAllWindows()
vs.release()