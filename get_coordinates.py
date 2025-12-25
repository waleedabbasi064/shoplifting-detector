import cv2

# Function to print coordinates on click
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked Point: [{x}, {y}]")
        # Draw a small circle so you know where you clicked
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Image', img)

# Read the video and get the first frame
video_path = 'gettyimages-1995820194-640_adpp.mp4'  # Make sure this matches your video file
cap = cv2.VideoCapture(video_path)
ret, img = cap.read()
cap.release()

if ret:
    print("Click on the 4 corners of your Danger Zone (Top-Left, Top-Right, Bottom-Right, Bottom-Left).")
    print("Press any key to close the window when done.")
    
    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Could not read video file. Check the path.")