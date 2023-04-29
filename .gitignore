import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
import os
import cv2
from keras.layers import Dense, Flatten

# Define the paths to your image folders
train_path = "C:\\Users\\bisoi\\OneDrive\\Desktop\\ML project\\train"
val_path = "C:\\Users\\bisoi\\OneDrive\\Desktop\\ML project\\val"

# Set the path to the folder containing the 'train' folder
data_dir = train_path

# Set the image size
img_size = (64, 64)

# Create empty lists for the images and labels
train_images = []
train_labels = []

# Loop over each folder from '0' to '9'
for label in range(10):
  folder_path = os.path.join(data_dir, str(label))
  # Loop over each image in the folder
  for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file)
    if file_path.endswith(('.jpg','.png')):
      # Load the image and resize it to the desired size
      img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
      img = cv2.resize(img, img_size)
      # Append the image and label to the lists
      train_images.append(img)
      train_labels.append(label)
      
# Convert the lists to NumPy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)
# Save the arrays in NumPy format
np.save('train_images.npy', train_images)
np.save('train_labels.npy', train_labels)

# Set the path to the folder containing the 'validation' folder
data_dir_val = val_path

# Set the image size
img_size_val = (64, 64)

# Create empty lists for the images and labels
val_images = []
val_labels = []

# Loop over each folder from '0' to '9'
for label in range(10):
  folder_path = os.path.join(data_dir_val, 'val', str(label))
  # Loop over each image in the folder
   for file in os.listdir(folder_path):
      file_path = os.path.join(folder_path, file)
      if file_path.endswith(('.png','.jpg')):
        # Load the image and resize it to the desired size
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size_val)
        # Append the image and label to the lists
        val_images.append(img)
        val_labels.append(label)
        
# Convert the lists to NumPy arrays
val_images = np.array(val_images)
val_labels = np.array(val_labels)

# Save the arrays in NumPy format
np.save('x_val.npy', val_images)
np.save('y_val.npy', val_labels)

# Load the training dataset
train_images = np.load('train_images.npy')
train_labels = np.load('train_labels.npy')

# Load the validation dataset
val_images = np.load('val_images.npy')
val_labels = np.load('val_labels.npy')

# Test that the images are loaded correctly
print("Number of training images:", len(x_train))
print("Number of testing images:", len(x_test))
print("Shape of first training image:", x_train[0].shape)
print("First training image:")
plt.matshow(x_train[0])
plt.show()
print("Random training image:")
plt.matshow(x_train[999])
plt.show()

print("Shape of training dataset:", x_train.shape)
print("Shape of testing dataset:", x_test.shape)
print("Labels of training dataset:", y_train)
print("Labels of testing dataset:", y_test)
print("Random testing image:")
plt.matshow(x_test[150])
plt.show()
