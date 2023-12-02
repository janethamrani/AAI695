import cv2
import os
import numpy as np

# Function to collect face images for training
def collect_faces():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Create a Haar Cascade Classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')    # Create a directory for storing face images
    face_dir = 'faces'
    if not os.path.exists(face_dir):
        os.makedirs(face_dir)

    # Counter for collected images
    count = 0

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Save the collected face images
            img_name = os.path.join(face_dir, f"face_{count}.png")
            cv2.imwrite(img_name, roi_color)

            # Draw a rectangle around the face
            color = (255, 0, 0)  # BGR format
            thickness = 2
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)

            count += 1

        # Display the collected images
        cv2.imshow('Collecting Faces', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 100:
            break

    cap.release()
    cv2.destroyAllWindows()

# Train the face recognizer model
def train_faces():
    face_dir = 'faces'
    images = []
    labels = []
    label_dict = {}
    current_id = 0

    # Loop through the collected images
    for root, dirs, files in os.walk(face_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace("face_", "").replace(" ", "-").lower()

                # Assign unique label ids to each person
                if not label in label_dict:
                    label_dict[label] = current_id
                    current_id += 1
                id_ = label_dict[label]

                # Convert images to grayscale and store them with labels
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_array = np.array(img, dtype=np.uint8)  # Ensure the correct data type

                # Append images and labels
                images.append(img_array)
                labels.append(id_)

    # Train the recognizer model using Eigenfaces
    recognizer = cv2.face.EigenFaceRecognizer_create()
    recognizer.train(images, np.array(labels))
    recognizer.save('face_trained.yml')  # Save the trained model to a file

    print("Training completed successfully!")

# Call the function to train the face recognition model
# Collect face images
collect_faces()

# Train the face recognizer model
train_faces()
