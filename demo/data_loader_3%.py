#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:48:39 2024

@author: nitaishah
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tifffile as tiff
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import keras
from keras import layers, models
import random
import shutil

tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})

# Disable any GPU-related issues by forcing the model to use CPU or other settings
tf.config.set_visible_devices([], 'GPU')

# Check available devices
print("Available devices:", tf.config.list_physical_devices())

# Path to the dataset
dataset_dir = '/Users/nitaishah/Desktop/ECEN-PROJECT/EuroSAT_MS/'  # Change this to your dataset path



class_labels = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]


data = []

for label, class_name in enumerate(class_labels):
    class_dir = os.path.join(dataset_dir, class_name)
    
    # Get all image filenames in the class directory
    all_images = os.listdir(class_dir)
    
    # Sample 20% of the images randomly
    sampled_images = random.sample(all_images, k=int(0.03 * len(all_images)))  # 20% sampling
    
    # Loop through the sampled images and add their paths and labels to the data list
    for image_name in sampled_images:
        image_path = os.path.join(class_dir, image_name)
        
        # Add image path, label, and class name to the data list
        data.append({"image_path": image_path, "label": label, "class_name": class_name})

# Convert the data list into a DataFrame
df = pd.DataFrame(data)

# Optionally, display the first few rows of the dataframe
print(df.head())

new_dataset_dir = '/Users/nitaishah/Desktop/ECEN-PROJECT/EuroSAT_MS_selected/'

# Create the new folder if it doesn't exist
if not os.path.exists(new_dataset_dir):
    os.makedirs(new_dataset_dir)

# Loop through each class
for label, class_name in enumerate(class_labels):
    class_dir = os.path.join(dataset_dir, class_name)
    
    # Get all image filenames in the class directory
    all_images = os.listdir(class_dir)
    
    # Sample 20% of the images randomly
    sampled_images = random.sample(all_images, k=int(0.03 * len(all_images)))  # 20% sampling
    
    # Create a folder for this class in the new dataset directory
    class_new_dir = os.path.join(new_dataset_dir, class_name)
    if not os.path.exists(class_new_dir):
        os.makedirs(class_new_dir)
    
    # Loop through the sampled images and move them to the new folder
    for image_name in sampled_images:
        image_path = os.path.join(class_dir, image_name)
        new_image_path = os.path.join(class_new_dir, image_name)
        
        # Move the image to the new directory
        shutil.copy(image_path, new_image_path)
        
        # Add image path, label, and class name to the data list
        data.append({"image_path": new_image_path, "label": label, "class_name": class_name})

# Convert the data list into a DataFrame
df = pd.DataFrame(data)

# Optionally, display the first few rows of the dataframe
print(df.head())


