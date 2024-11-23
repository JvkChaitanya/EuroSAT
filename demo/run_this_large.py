#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:43:41 2024

@author: nitaishah
"""

from Image_Functions import *
from keras.models import load_model
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})

tf.config.set_visible_devices([], 'GPU')

print("Available devices:", tf.config.list_physical_devices())


dataset_dir = 'EuroSAT_MS_selected'  # Update this path to where your EuroSat_selected folder is located


class_labels = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]


image_data = []


for class_idx, class_name in enumerate(class_labels):
    class_folder = os.path.join(dataset_dir, class_name)
    
    # Ensure the folder exists
    if os.path.exists(class_folder):
        # Loop through each image in the class folder
        for image_name in os.listdir(class_folder):
            image_path = os.path.join(class_folder, image_name)
            
            # Ensure it's a file (and not a directory)
            if os.path.isfile(image_path):
                # Append the image path and corresponding ground truth label (class_idx) to the list
                image_data.append({"image_path": image_path, "label": class_idx, "class_name": class_name})

df_images = pd.DataFrame(image_data)

def convert_image_to_array(image_path):
    array = read_image_as_array(image_path)
    array = array.transpose(1,2,0)
    array = array/np.max(array)
    array = np.expand_dims(array, axis=0) 
    return array


model_path = "model_1.h5"
model = load_model(model_path)

def predict_and_store_results(df_images, model):

    predictions_data = []
    
    for _, row in df_images.iterrows():
        image_path = row['image_path']
        true_label = row['label']
        true_class_name = row['class_name']
        
        image_array = convert_image_to_array(image_path)
        
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions) 
        
        predictions_data.append({
            "image_path": image_path,
            "true_label": true_label,
            "true_class_name": true_class_name,
            "predicted_class": predicted_class,
            "predicted_class_name": class_labels[predicted_class],
            "prediction_probabilities": predictions.flatten()
        })
    

    df_predictions = pd.DataFrame(predictions_data)
    return df_predictions


df_predictions = predict_and_store_results(df_images, model)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
def evaluate_model(y_truth, y_label):
    
    accuracy = accuracy_score(y_truth, y_label)
    
    
    precision = precision_score(y_truth, y_label, average='micro')
    recall = recall_score(y_truth, y_label, average='micro')
    f1 = f1_score(y_truth, y_label, average='micro')
    
    # Print the metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    cm = confusion_matrix(y_truth.flatten(), y_label.flatten())
    
    unique_class_names = df['class_name'].unique()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=unique_class_names,
        yticklabels=unique_class_names
    )
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix with Class Names')
    plt.show()   


evaluate_model(np.array(df_predictions['true_label']), np.array(df_predictions['predicted_class']))



