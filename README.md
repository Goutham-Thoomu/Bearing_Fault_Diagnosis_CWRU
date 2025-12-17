# Bearing_Fault_Diagnosis_CWRU
This project focuses on bearing fault classification using vibration signal data from the Case Western Reserve University (CWRU) bearing dataset.  
Both classical machine learning (XGBoost) and deep learning (1D CNN) models are implemented and compared.

# Problem Statement
Bearing failures are a major cause of unplanned downtime in rotating machinery.
Early fault detection helps reduce maintenance cost and improve system reliability.
In this project, vibration signals are used to classify different bearing conditions such as:
1) Normal
2) Inner race fault
3) Outer race fault
4) Ball fault

# Dataset Information 
Dataset: Case Western Reserve University (CWRU) Bearing Dataset
Signal type: Vibration data from accelerometers
Sampling frequency: 12 kHz / 48 kHz
Fault sizes: 0.007", 0.014", 0.021"
Dataset source: https://engineering.case.edu/bearingdatacenter/download-data-file 

# Data Processing
The following steps were performed on the raw vibration signals: 
1) Segmentation of time-domain signals into fixed-length windows
2) Fast Fourier Transform (FFT) to convert signals to frequency domain
3) Feature extraction for machine learning models
4) Data normalization and labeling

# Models Used 
#1. XGBoost Classifier
Used FFT-based statistical features
Suitable for structured feature-based learning
Fast training and good performance on tabular data 

##2. 1D Convolutional Neural Network (CNN)
Takes vibration signal segments as input 
Learns features automatically from the data 
Designed for time-series clssification

# Results
XGBoost MODEL ACCURACY: 0.9690996270644646
1D CNN Test Accuracy:  0.9978689551353455 

# Key Learnings
Difference between feature based ML model and deep learning approaches
Importance of signal preprocessing in vibration analysis
Trade offs between model accuracy and computational complexity
Practical understanding of undustrial fault diagnosis problems

# Tools & Technologies 
Python
Numpy, SciPy
Pandas
Scikit-learn
XGboost
Tensorflow
Matplotlib
Seaborn
