import cv2

url = "rtsp://user:pass@192.168.1.10:554/stream"  # or http://...

cam = cv2.VideoCapture(url)  # sometimes: cv2.VideoCapture(url, cv2.CAP_FFMPEG)

if not cam.isOpened():
    raise RuntimeError("Failed to open stream")

ret, frame = cam.read()
if not ret:
    raise RuntimeError("Failed to read first frame")

h, w = frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (w, h))

while True:
    ret, frame = cam.read()
    if not ret:
        break  # stream ended / dropped

    out.write(frame)
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
out.release()
cv2.destroyAllWindows()