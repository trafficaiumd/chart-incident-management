import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
model=YOLO("/home/group1/chart-incident-management/YOLO Testing Ariyan/Practice/CODE 2/weights/epoch14.pt")
#results = model("/home/group1/chart-incident-management/YOLO Testing Ariyan/Practice/CODE 2/testing/istockphoto.mp4", show=True, conf=0.7)
results = model("/home/group1/chart-incident-management/YOLO Testing Ariyan/Practice/CODE 2/testing/fig1.jpg", show=True, conf=0.7)

# If you want to visualize the result inside notebook (optional):
img_with_boxes = results[0].plot()

plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Detections")
plt.show()

#Between 24 to 25.5 FPS