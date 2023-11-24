import cv2
import os

def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def capture_faces(cascade, frame, face_count, save_path, max_faces=100):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        # Increase the height of the rectangle
        height_increase = 20  # Adjust this value to change the height of the rectangle
        y_new = y - height_increase // 2
        h_new = h + height_increase

        # Draw a rectangle with increased height
        cv2.rectangle(frame, (x, y_new), (x + w, y_new + h_new), (255, 0, 0), thickness=3)

        face_region = frame[y:y+h, x:x+w]
        if face_count < max_faces:
            cv2.imwrite(os.path.join(save_path, f'face_{face_count}.jpg'), face_region)
            face_count += 1
    return frame, face_count



def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    save_path = r'C:\Users\suyas\OneDrive-stevens.edu\Desktop\Applied Machine Learning\Final_Project\images'
    create_directory(save_path)
    face_count = 0
    max_faces = 100

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, face_count = capture_faces(face_cascade, frame, face_count, save_path, max_faces)

        cv2.imshow('Capture Faces', frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or face_count >= max_faces:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
