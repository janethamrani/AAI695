from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os

# Define your data augmentation generator
data_gen = ImageDataGenerator(
    rotation_range=20,          # Randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,      # Randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,     # Randomly shift images vertically (fraction of total height)
    rescale=1./255,             # Rescale the image by normalizing it.
    shear_range=0.2,            # Shear angle in counter-clockwise direction (degrees)
    zoom_range=0.2,             # Randomly zoom image 
    horizontal_flip=True,       # Randomly flip images horizontally
    fill_mode='nearest'         # Set the strategy used for filling in newly created pixels
)

# Assuming you have images for training in a folder named 'dataset/train'
# and the images are grouped in subfolders by class (e.g., 'faces', 'no_faces')
train_data_dir = r'C:\Users\suyas\OneDrive-stevens.edu\Desktop\AAI695\AAI695\images'

# The directory where you want to save the augmented images
save_dir = r'C:\Users\suyas\OneDrive-stevens.edu\Desktop\AAI695\AAI695\images\augmented_images'

# Iterate over each image and transform it
for img_file in os.listdir(train_data_dir):
    img_path = os.path.join(train_data_dir, img_file)
    img = load_img(img_path)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 256, 256)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 256, 256)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `save_dir` directory
    i = 0
    for batch in data_gen.flow(x, batch_size=1, save_to_dir=save_dir, save_prefix='aug', save_format='jpeg'):
        i += 1
        if i > 20:  # this loop is just to get one sample of the transformations
            break
