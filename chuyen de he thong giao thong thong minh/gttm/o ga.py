# su dung thuat toan yolov4 tiny
import cv2
import os
#doc anh
img = cv2.imread("D:/IT/gttm/2.jpg")
#doc cac  obj.name chua cac danh sach doi tuong
with open(os.path.join("D:/IT/gttm/project_files", 'obj.names'), 'r') as f:
    classes = f.read().splitlines()
#nap mo hinh yolov4 tiny voi cac trong so va tiep cau hinh
net = cv2.dnn.readNet('D:/IT/gttm/project_files/yolov4_tiny.weights', 'D:/IT/gttm/project_files/yolov4_tiny.cfg')
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
#nhan dien cac doi tuong trong anh
classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)
#thuc hien ve hinh chu nhat trong buc anh
for (classId, score, box) in zip(classIds, scores, boxes):
    cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                  color=(0, 255, 0), thickness=2)
#hien thi hinh anh va luu ket qua
cv2.imshow("pothole",img)
cv2.imwrite("result1"+".jpg",img) #result name
cv2.waitKey(0)
cv2.destroyAllWindows()