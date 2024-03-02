import cv2
from random import randrange

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()
    gray_style_picture = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    face_coordinates = face_cascade.detectMultiScale(gray_style_picture)
    
    # Draw rectangles around the detected faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(0, 255), randrange(0, 255), randrange(0, 255)), 7)
    
    # Display the frame with detected faces
    cv2.imshow('Face Detection', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
