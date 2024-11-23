# EuroSAT
The EuroSAT dataset includes imagery categorised into 10 distinct land cover classes: Industrial, Residential, Annual Crop, Permanent Crop, River, Sea & Lake, Herbaceous Vegetation, Highway, Pasture, and Forest. Each class contains a varying number of images corresponding to its specific category, collectively amounting to a total of 27,000 images. These images are derived from Sentinel-2 satellite data and span 13 spectral bands, enabling a comprehensive analysis of land cover types. The 13 spectral bands include: Coastal Aerosol, Blue, Green, Red, Red Edge 1, Red Edge 2, Red Edge 3, Near Infrared (NIR), Narrow Near Infrared, Water Vapor, Cirrus, Shortwave Infrared 1 (SWIR1) and Shortwave Infrared 2 (SWIR2)
File Structure -- 

demo - This folder contains our baseline model trained on multispectral data saved as a h5 file. We have provided the file run_this.py that loads 1 image from each class present in the folder images_test, and applies the model to it. Models.py contains the code we used to train our model. Image functions.py contains several key functions we reference in our code. run_this_large.py applies our model on roughly 20% of the total data. 

RGB_MODEL - This folder contains our model trained on RGB data saved as a h5 file. We have provided the file run_this.py that loads 1 image from each class present in the folder images_test, and applies the model to it. rgb_only.py contains the code we used to train our model. Image functions.py contains several key functions we reference in our code.

NDVI - This folder contains the code for NDVI_calculation. To see this run run_this.py. In case the outputs are not visible, we have provided an output folder that contains the results 

