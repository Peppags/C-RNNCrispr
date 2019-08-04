# C-RNNCrispr 
## Overview
C-RNNCrispr is a deep learning-based method for CRISPR/Cas9 guide RNA (gRNA) on-target cleavage efficacy prediction.

## Pre-requisite:  
* **Ubuntu 16.04**
* **Anaconda 3-5.2.0**
* **Python packages:**   
  [numpy](https://numpy.org/) 1.16.4  
  [pandas](https://pandas.pydata.org/) 0.23.0  
  [scikit-learn](https://scikit-learn.org/stable/) 0.19.1  
  [scipy](https://www.scipy.org/) 1.1.0  
 * **[Keras](https://keras.io/) 1.1.0**    
 * **Tensorflow and dependencies:**   
  [Tensorflow](https://tensorflow.google.cn/) 1.4.0    
  CUDA 8.0 (for GPU use)    
  cuDNN 6.0 (for GPU use)    
  
## Installation guide
#### **Operation system**  
Ubuntu 16.04 download from https://www.ubuntu.com/download/desktop  
#### **Python and packages**  
Download Anaconda 3-5.2.0 tarball on https://www.anaconda.com/distribution/#download-section  
#### **Tensorflow installation:**  
pip install tensorflow-gpu==1.4.0 (for GPU use)  
pip install tensorflow==1.4.0 (for CPU use)  
#### **CUDA toolkit 8.0 (for GPU use)**     
Download CUDA tarball on https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run  
#### **cuDNN 6.1.10 (for GPU use)**      
Download cuDNN tarball on https://developer.nvidia.com/cudnn  

## Content
* **./data:** the testing examples with sgRNA sequence and corresponding epigenetic features and label indicating the on-target cleavage efficacy  
* **./weights/C_RNNCrispr_weights.h5:** the well-trained weights for our model    
* **./C_RNNCrispr_test.py:** the python code, it can be ran to reproduce our results  

