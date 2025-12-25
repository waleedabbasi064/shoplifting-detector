import cv2
# Create 'get_frame.py'
cap = cv2.VideoCapture('gettyimages-1995820194-640_adpp.mp4')
ret, frame = cap.read()
cv2.imwrite('zone_reference.jpg', frame)
print("Saved zone_reference.jpg. Download it and find your coordinates!")
cap.release()