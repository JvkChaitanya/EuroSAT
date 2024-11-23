# EuroSAT
The EuroSAT dataset includes imagery categorised into 10 distinct land cover classes: Industrial, Residential, Annual Crop, Permanent Crop, River, Sea & Lake, Herbaceous Vegetation, Highway, Pasture, and Forest. Each class contains a varying number of images corresponding to its specific category, collectively amounting to a total of 27,000 images. These images are derived from Sentinel-2 satellite data and span 13 spectral bands, enabling a comprehensive analysis of land cover types. The 13 spectral bands include: Coastal Aerosol, Blue, Green, Red, Red Edge 1, Red Edge 2, Red Edge 3, Near Infrared (NIR), Narrow Near Infrared, Water Vapor, Cirrus, Shortwave Infrared 1 (SWIR1) and Shortwave Infrared 2 (SWIR2)
File Structure -- 

demo - This folder contains our baseline model trained on multispectral data saved as a h5 file. We have provided the file run_this.py that loads 1 image from each class present in the folder images_test, and applies the model to it.Models.py contains the code we used to train our model. Image functions.py contains several key functions we reference in our code. run_this_large.py applies our model on roughly 3% of the total data and displays eval metrics along with the confusion matrix. Data_Loader.py is a function we wrote to choose 3% of the data to run run_this_large.py. We chose this value since Github does not allow for files larger than 100 MB to be uploaded. 

RGB_MODEL - This folder contains our model trained on RGB data saved as a h5 file. We have provided the file run_this.py that loads 1 image from each class present in the folder images_test, and applies the model to it. rgb_only.py contains the code we used to train our model. Image functions.py contains several key functions we reference in our code.

NDVI - This folder contains the code for NDVI_calculation. To see this run run_this.py. In case the outputs are not visible, we have provided an output folder that contains the results 

Segmentation Masking - This folder contains the code we use to apply our pixel wise classification models to the Geo-FS inputs. We have provided the weights file, along with the file to train the model. To run this model use run_this.py. The outputs are provided in the output directory 

Zero Shot Classification - This folder contains the code we use to apply Zero Shot Classification to our models. run_this.py loads images from the images_test folder and applies them to the images. zero-shot-classification.py contains the code for the entire data instead of a subset. 

EDA - This folder contains the code to perfom eda on our data. run_this.py loads the test images, and displays class metrics, and also displays different band combinations present in the multispectral data. 

Conv3D - This folder contains the code we use to apply classification using Conv3D models. demo.py loads images from the test folder and applies them to the images. train_model_conv3D.ipynb contains the code to train the model.(change directory and model paths accordingly before running the code files)
