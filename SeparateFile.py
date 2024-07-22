import os
import shutil
import random

# Set the path to the main folder containing all the images
main_folder = "C:/Users/user/Desktop/Degree/Y2S3/UECS3413 Digital Image Processing/practical/DIPASGN/DIPDataSet"

# Define the ratios for train, test, and validation sets
train_ratio = 0.6
test_ratio = 0.2
valid_ratio = 0.2

# Define the paths for the train, test, and validation folders
train_folder = "C:/Users/user/Desktop/Degree/Y2S3/UECS3413 Digital Image Processing/practical/DIPASGN/Train"
test_folder = "C:/Users/user/Desktop/Degree/Y2S3/UECS3413 Digital Image Processing/practical/DIPASGN/Test"
valid_folder = "C:/Users/user/Desktop/Degree/Y2S3/UECS3413 Digital Image Processing/practical/DIPASGN/Valid"

# Create the train, test, and validation folders 
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
os.makedirs(valid_folder, exist_ok=True)

# Iterate through each class folder in the main folder
for class_folder in os.listdir(main_folder):
    class_path = os.path.join(main_folder, class_folder)
    if os.path.isdir(class_path):
        # Get the list of images in the class folder
        images = os.listdir(class_path)
        # Shuffle the list of images randomly
        random.shuffle(images)
        
        # Calculate the number of images for each set
        num_train = int(len(images) * train_ratio)
        num_test = int(len(images) * test_ratio)
        num_valid = len(images) - num_train - num_test
        
        # Split the images into train, test, and validation sets
        train_images = images[:num_train]
        test_images = images[num_train:num_train + num_test]
        valid_images = images[num_train + num_test:]
        
        # Move the images to their respective folders
        for image in train_images:
            src = os.path.join(class_path, image)
            dest = os.path.join(train_folder, class_folder, image)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(src, dest)
        for image in test_images:
            src = os.path.join(class_path, image)
            dest = os.path.join(test_folder, class_folder, image)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(src, dest)
        for image in valid_images:
            src = os.path.join(class_path, image)
            dest = os.path.join(valid_folder, class_folder, image)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(src, dest)
