import cv2
import os

# Define a function to create a directory if it doesn't exist
def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# Define a function to detect eyes within a face region
def detect_eyes_in_face(eye_cascade, face_region):
    # Detect eyes within the face region using the provided cascade classifier
    eyes = eye_cascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Return True if at least one eye is detected, which helps in validating the face detection
    return len(eyes) >= 1

# Define a function to capture faces from a video frame
def capture_faces(face_cascade, eye_cascade, frame, face_count, save_path, max_faces=100):
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract the region of interest in grayscale and color from the face
        face_region_gray = gray[y:y+h, x:x+w]
        face_region_color = frame[y:y+h, x:x+w]
        
        # Check if eyes are detected in the face region
        if detect_eyes_in_face(eye_cascade, face_region_gray):
            # If eyes are detected and the face count is less than the maximum, save the face image
            if face_count < max_faces:
                # Save the detected face region
                cv2.imwrite(os.path.join(save_path, f'face_{face_count}.jpg'), face_region_color)
                # Increment the face count
                face_count += 1
                # Draw a rectangle around the detected face in the frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), thickness=2)
    # Return the frame with drawn rectangles and the updated face count
    return frame, face_count

def main():
    # Load the cascades for face and eye detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    # Start the webcam
    cap = cv2.VideoCapture(0)

    # Define the save path for the captured images
    save_path = r'C:\Users\suyas\OneDrive-stevens.edu\Desktop\AAI695\AAI695\images'
    # Create the directory if it doesn't exist
    create_directory(save_path)
    # Initialize the face count and set the maximum number of faces to capture
    face_count = 0
    max_faces = 100

    # Main loop to capture video frames
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame to capture faces
        frame, face_count = capture_faces(face_cascade, eye_cascade, frame, face_count, save_path, max_faces)

        # Display the frame with detected faces
        cv2.imshow('Capture Faces', frame)

        # Break the loop if 'q' is pressed or the maximum number of faces is reached
        if cv2.waitKey(1) & 0xFF == ord('q') or face_count >= max_faces:
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
