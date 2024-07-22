import os
import cv2  # OpenCV
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, concatenate
from keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Define paths to train, test, and validation folders
#Change the current_directory to location of your folder if you guys want to try to run it
current_directory = os.getcwd()
train_folder = os.path.join(current_directory, "Train")
test_folder = os.path.join(current_directory, "Test")
valid_folder = os.path.join(current_directory, "Valid")

# Define image dimensions and batch size
img_width, img_height = 150, 150
batch_size = 32

# Function to load and preprocess images from a folder
def load_images(folder_path):
    images = []
    labels = []
    label_to_index = {}  # Map class names to numerical labels
    for class_index, class_name in enumerate(os.listdir(folder_path)):
        class_folder = os.path.join(folder_path, class_name)
        if os.path.isdir(class_folder):
            label_to_index[class_name] = class_index
            for file_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, file_name)
                if image_path.endswith(".jpg") or image_path.endswith(".png"):
                    # Load image using OpenCV (you can also use PIL)
                    img = cv2.imread(image_path)
                    if img is not None:  # Check if image loading was successful
                        # Resize image to target dimensions
                        img = cv2.resize(img, (img_width, img_height))
                        if img.shape[0] != 0 and img.shape[1] != 0:  # Check if the resized image is not empty
                            # Preprocess image (e.g., convert to float and normalize)
                            img = img.astype(np.float32) / 255.0
                            images.append(img)
                            labels.append(class_name)
    # Convert class names to numerical labels
    numerical_labels = [label_to_index[label] for label in labels]
    return np.array(images), np.array(numerical_labels)

# Load and preprocess images from train, test, and validation folders
X_train, Y_train = load_images(train_folder)
X_test, Y_test = load_images(test_folder)
X_valid, Y_valid = load_images(valid_folder)

# One-hot encode labels using tf.one_hot
num_classes = len(np.unique(Y_train))
Y_train = tf.one_hot(Y_train, depth=num_classes)
Y_test = tf.one_hot(Y_test, depth=num_classes)
Y_valid = tf.one_hot(Y_valid, depth=num_classes)

def inception_module(x, filters):
    # 1x1 convolution
    conv1x1 = Conv2D(filters=filters[0], kernel_size=(1, 1), padding='same', activation='relu')(x)
    
    # 3x3 convolution
    conv3x3 = Conv2D(filters=filters[1], kernel_size=(3, 3), padding='same', activation='relu')(x)
    
    # 5x5 convolution
    conv5x5 = Conv2D(filters=filters[2], kernel_size=(5, 5), padding='same', activation='relu')(x)
    
    # Max pooling
    maxpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    maxpool_conv = Conv2D(filters=filters[3], kernel_size=(1, 1), padding='same', activation='relu')(maxpool)
    
    # Concatenate the outputs
    output = concatenate([conv1x1, conv3x3, conv5x5, maxpool_conv], axis=-1)
    return output

def create_googlenet():
    input_layer = Input(shape=(img_width, img_height, 3))
    
    # First Convolutional Layer
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    # Second Convolutional Layer
    x = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    x = Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    # Inception Module 1
    x = inception_module(x, [64, 128, 32, 32])
    x = inception_module(x, [128, 192, 96, 64])
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    # Flatten and Fully Connected Layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Create the GoogLeNet model
model = create_googlenet()

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, Y_train,
    batch_size=batch_size,
    epochs=10,
    validation_data=(X_valid, Y_valid))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Predictions on test set
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true_classes = np.argmax(Y_test, axis=1)

# Calculate classification metrics
accuracy = accuracy_score(Y_true_classes, Y_pred_classes)
precision = precision_score(Y_true_classes, Y_pred_classes, average='weighted')
recall = recall_score(Y_true_classes, Y_pred_classes, average='weighted')
f1 = f1_score(Y_true_classes, Y_pred_classes, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
