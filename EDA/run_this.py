#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 01:12:39 2024

@author: nitaishah
"""

import Image_Functions
from Image_Functions import *
import seaborn as sns

dataset_dir = 'images_test/'  # Change this to your dataset path



class_labels = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]


data = []


for label, class_name in enumerate(class_labels):
    class_dir = os.path.join(dataset_dir, class_name)
    
    # Loop through each image in the class directory
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        
        # Add image path and label to the data list
        data.append({"image_path": image_path, "label": label, "class_name": class_name})


df = pd.DataFrame(data)

class_counts = df['class_name'].value_counts()

df['image_path'][0]

for i in range(0, 10):
    multispectral_array = read_image_as_array(df['image_path'][i])
    display_rgb_from_array(multispectral_array)
    display_nir_composite(multispectral_array)
    display_ndvi(multispectral_array)
    display_false_color_infrared(multispectral_array)
    display_swir_composite(multispectral_array)
    display_urban_composite(multispectral_array)
    display_moisture_index_composite(multispectral_array)
    display_geology_composite(multispectral_array)
    display_ndwi(multispectral_array)
    display_atmospheric_composite(multispectral_array)

