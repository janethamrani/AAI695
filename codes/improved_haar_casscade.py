import cv2
import os

def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def detect_eyes_in_face(eye_cascade,face_region):
    eyes = eye_cascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(eyes) >= 1  # Returns True if at least one eye is detected

def capture_faces(face_cascade, eye_cascade, frame, face_count, save_path, max_faces=100):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        face_region_gray = gray[y:y+h, x:x+w]
        face_region_color = frame[y:y+h, x:x+w]
        
        # Check for eyes in the face region to validate detection
        if detect_eyes_in_face(eye_cascade, face_region_gray):
            # If eyes are detected, save the face and draw a rectangle
            if face_count < max_faces:
                cv2.imwrite(os.path.join(save_path, f'face_{face_count}.jpg'), face_region_color)
                face_count += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), thickness=2)
    return frame, face_count

def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    cap = cv2.VideoCapture(0)

    save_path = r'C:\Users\suyas\OneDrive-stevens.edu\Desktop\Applied Machine Learning\Final_Project\images'
    create_directory(save_path)
    face_count = 0
    max_faces = 100

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, face_count = capture_faces(face_cascade, eye_cascade, frame, face_count, save_path, max_faces)

        cv2.imshow('Capture Faces', frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or face_count >= max_faces:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
