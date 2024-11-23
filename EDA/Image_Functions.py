#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 01:06:10 2024

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

def read_image_as_array(file_path):
    with rasterio.open(file_path) as src:
    # Reading all 13 bands
        coastal_aerosol = src.read(1)      # Band 1 (443 nm)
        blue = src.read(2)                 # Band 2 (490 nm)
        green = src.read(3)                # Band 3 (560 nm)
        red = src.read(4)               # Band 4 (665 nm)
        vegetation_red_edge_1 = src.read(5)  # Band 5 (705 nm)
        vegetation_red_edge_2 = src.read(6)  # Band 6 (740 nm)
        vegetation_red_edge_3 = src.read(7)  # Band 7 (783 nm)
        nir = src.read(8)                  # Band 8 (842 nm)
        narrow_nir = src.read(9)           # Band 8A (865 nm)
        water_vapor = src.read(10)         # Band 9 (945 nm)
        cirrus = src.read(11)              # Band 10 (1375 nm)
        swir1 = src.read(12)               # Band 11 (1610 nm)
        swir2 = src.read(13)               # Band 12 (2190 nm)
    
        # Optional: Stack into a single array of shape (13, H, W)
        multispectral_array = src.read(list(range(1, 14)))  # Reads all bands at once
        
        return multispectral_array




def display_rgb_from_array(array):
    """
    Display an RGB composite from a multispectral array.

    Parameters:
    array (numpy.ndarray): A 3D array with shape (bands, height, width).
                           The array is expected to have at least 4 bands,
                           where band indices for RGB are (Red=3, Green=2, Blue=1).
    """
    # Check if array has the expected shape (bands, height, width)
    if array.shape[0] < 4:
        raise ValueError("Input array must have at least 4 bands for RGB display.")

    # Extract the Red, Green, and Blue bands (assuming Band 4=Red, Band 3=Green, Band 2=Blue)
    red = array[3]
    green = array[2]
    blue = array[1]


    rgb = np.dstack((red, green, blue))

  
    rgb = rgb / np.max(rgb)


    plt.imshow(rgb)
    plt.axis('off')
    plt.show()

def display_nir_composite(array):
    if array.shape[0] < 5:
        raise ValueError("Input array must have at least 5 bands for NIR composite display.")
    
    nir = array[4]
    green = array[2]
    blue = array[1]
    nir_composite = np.dstack((nir, green, blue)) / np.max(array)
    
    plt.imshow(nir_composite)
    plt.axis('off')
    plt.show()

def display_ndvi(array):
    if array.shape[0] < 5:
        raise ValueError("Input array must have at least 5 bands for NDVI calculation.")
    
    nir = array[4].astype(float)
    red = array[3].astype(float)
    ndvi = (nir - red) / (nir + red)
    
    plt.imshow(ndvi, cmap='RdYlGn')
    plt.colorbar(label="NDVI")
    plt.axis('off')
    plt.show()

def display_false_color_infrared(array):
    if array.shape[0] < 5:
        raise ValueError("Input array must have at least 5 bands for false color infrared display.")
    
    nir = array[4]
    red = array[3]
    green = array[2]
    false_color_infrared = np.dstack((nir, red, green)) / np.max(array)
    
    plt.imshow(false_color_infrared)
    plt.axis('off')
    plt.show()
    
import numpy as np
import matplotlib.pyplot as plt

def display_false_color_composite(array):
    
    # Check if array has the expected shape

    # Extract the NIR, Red, and Green bands for FCC
    nir = array[4]
    red = array[3]
    green = array[2]

    # Stack bands into an FCC image
    fcc = np.dstack((nir, red, green))

    # Normalize the FCC values to be in the range 0-1 for display purposes
    fcc = fcc / np.max(fcc)

    # Display the FCC image
    plt.imshow(fcc)
    plt.axis('off')
    plt.show()
    
def display_swir_composite(array):
    if array.shape[0] < 12:
        raise ValueError("Input array must have at least 12 bands for SWIR composite display.")
    
    swir1 = array[11]
    nir = array[8]
    red = array[4]
    swir_composite = np.dstack((swir1, nir, red)) / np.max(array)
    
    plt.imshow(swir_composite)
    plt.axis('off')
    plt.show()
    
def display_urban_composite(array):
    if array.shape[0] < 13:
        raise ValueError("Input array must have at least 13 bands for Urban composite display.")
    
    swir2 = array[12]
    nir = array[8]
    red = array[4]
    urban_composite = np.dstack((swir2, nir, red)) / np.max(array)
    
    plt.imshow(urban_composite)
    plt.axis('off')
    plt.show()

def display_moisture_index_composite(array):
    if array.shape[0] < 12:
        raise ValueError("Input array must have at least 12 bands for Moisture Index composite display.")
    
    swir1 = array[11]
    nir = array[8]
    green = array[3]
    moisture_index = np.dstack((swir1, nir, green)) / np.max(array)
    
    plt.imshow(moisture_index)
    plt.axis('off')
    plt.show()

def display_geology_composite(array):
    if array.shape[0] < 13:
        raise ValueError("Input array must have at least 13 bands for Geology composite display.")
    
    swir2 = array[12]
    swir1 = array[11]
    red = array[4]
    geology_composite = np.dstack((swir2, swir1, red)) / np.max(array)
    
    plt.imshow(geology_composite)
    plt.axis('off')
    plt.show()

def display_ndwi(array):
    if array.shape[0] < 9:
        raise ValueError("Input array must have at least 9 bands for NDWI calculation.")
    
    green = array[3].astype(float)
    nir = array[8].astype(float)
    ndwi = (green - nir) / (green + nir)
    
    plt.imshow(ndwi, cmap='Blues')
    plt.colorbar(label="NDWI")
    plt.axis('off')
    plt.show()


def display_atmospheric_composite(array):
    if array.shape[0] < 11:
        raise ValueError("Input array must have at least 11 bands for Atmospheric composite display.")
    
    cirrus = array[10]
    coastal_aerosol = array[1]
    blue = array[2]
    atmospheric_composite = np.dstack((cirrus, coastal_aerosol, blue)) / np.max(array)
    
    plt.imshow(atmospheric_composite)
    plt.axis('off')
    plt.show()
