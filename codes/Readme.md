'''
This is the Readme file for codes ,Call Stack and its API design .
Ex:
main.py---->image_capture.py---> so on..


main.py :this program is used for ...
image_capture.py :this program uses ...
'''
_____ Readme starts here _____
main.py : this  program contains the facial recognition algorithm to recognise faces from the profile 
improved_haar_casscade.py : this program detects eye/faces using the webcam
augment.py : this program shuffles the images for better training 
multi_face_detect.py : this program detects multiple faces in the webcam and stores different images in different subfolders which will help to create a profile
multi_face_detect.py ---->augment.py ---->main.py

### Multiface code description
This code integrates OpenCV and face_recognition libraries to develop a real-time face recognition system utilizing a webcam feed. The script leverages pre-trained face detection models from OpenCV and the face_recognition library, encapsulating functions for directory creation, face encoding, and matching. Upon initializing the webcam, the program continuously captures frames, converts them to grayscale for face detection, and encodes the detected faces. Detected faces are compared against a list of known face encodings; if a match is not found within a defined tolerance, the system creates a new folder for the unrecognized face, saves the face image, and updates the database of known faces. Notably, the script illustrates the process of recognizing faces, marking them with bounding rectangles, and storing identified faces in separate directories.

### Augment code
This code showcases image augmentation techniques implemented using the TensorFlow Keras library. It utilizes the ImageDataGenerator class to perform a variety of augmentations on a collection of images. The augmentation parameters include rotation within a defined range, horizontal and vertical shifts, rescaling for normalization, shearing, zooming, and horizontal flipping. The script accesses a directory containing images grouped by class (e.g., 'faces', 'no_faces') within the 'dataset/train' folder. It applies the defined augmentation techniques to each image and saves the augmented images into a specified directory named 'augmented_images'. The code iterates through each image in the provided directory, converts it into a Numpy array, and then applies transformations using the ImageDataGenerator's .flow() method. It generates multiple augmented versions of each image, ensuring a diverse training dataset.
