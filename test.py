import cv2

# Try indexes from 0 to 5
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera index {i} is available")
        cap.release()
