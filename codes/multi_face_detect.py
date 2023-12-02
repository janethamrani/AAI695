import cv2
import face_recognition
import os
import numpy as np

# Function to create a directory if it does not exist
def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# Function to find a matching face encoding within a given tolerance
def find_matching_face_encoding(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding_to_check, tolerance)
    if True in matches:
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding_to_check)
        best_match_index = np.argmin(face_distances)
        return best_match_index
    return None

# Main function
def main():
    # Load the pre-trained model for face detection from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Start the webcam
    cap = cv2.VideoCapture(0)

    # Directory where images will be saved
    base_save_path = r'C:\Users\suyas\OneDrive-stevens.edu\Desktop\AAI695\AAI695\images'
    create_directory(base_save_path)

    known_face_encodings = []  # List to store known face encodings
    known_faces_folders = []   # List to store directories for known faces
    face_id = 0                # Counter for assigning IDs to faces

    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            break

        # Convert the frame to RGB and grayscale for face detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Define the face location
            face_location = (y, x+w, y+h, x)
            # Encode the face
            face_encodings = face_recognition.face_encodings(rgb_frame, [face_location])

            if face_encodings:
                face_encoding = face_encodings[0]
                match_index = find_matching_face_encoding(known_face_encodings, face_encoding)

                if match_index is None:
                    # Create a new folder for a new face
                    face_folder = os.path.join(base_save_path, f'face_{face_id}')
                    create_directory(face_folder)
                    known_faces_folders.append(face_folder)
                    known_face_encodings.append(face_encoding)
                    match_index = face_id
                    face_id += 1

                # Save the face image in the corresponding folder
                face_image = frame[y:y+h, x:x+w]
                face_count = len(os.listdir(known_faces_folders[match_index]))
                cv2.imwrite(os.path.join(known_faces_folders[match_index], f'{face_count}.jpg'), face_image)

            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), thickness=2)

        # Display the frame with detected faces
        cv2.imshow('Capture Faces', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the main function if this script is executed
if __name__ == '__main__':
    main()
