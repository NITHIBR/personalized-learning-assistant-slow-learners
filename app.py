import pandas as pd
import face_recognition
import cv2


# Load an image with known faces
known_image = face_recognition.load_image_file("download (1).jpg")

# Encode the known faces
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# List of known face encodings and their corresponding labels
known_face_encodings = [known_face_encoding]
known_face_labels = ["SRK"]

# Load an image for face recognition
unknown_image = face_recognition.load_image_file("download (1).jpg")

# Find face locations in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Initialize an empty list to store names of recognized faces
face_names = []

# Loop through each face found in the unknown image
for face_encoding in face_encodings:
    # Compare the face encoding of the unknown face with the known face encodings
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    # Check if the unknown face matches any known faces
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_labels[first_match_index]

    face_names.append(name)

# Display the results
for (top, right, bottom, left), name in zip(face_locations, face_names):
    cv2.rectangle(unknown_image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(unknown_image, name, (left, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

# Display the image
cv2.imwrite("output_image.jpg", unknown_image)

cv2.destroyAllWindows()