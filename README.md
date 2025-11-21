
# Sign Language Detection (CV Project)

## Project title
*Sign Language Detection (Computer Vision)*

## Overview of the project
![Demo GIF]
This project uses MediaPipe hand landmarks + OpenCV to capture hand sign images, create a dataset of normalized hand landmarks, train a RandomForest classifier on those landmarks, and run real-time predictions with the webcam.

The workflow:
1. Capture images for each sign using the webcam (collect_imgs.py).
   
## Demo video
Watch the full demo on how to capture the data :
https://drive.google.com/file/d/1Q4d-tTJtkU5jnwB-6m7SN8JmQUEBnS0w/view?usp=sharing

2. Convert images to landmark-based dataset (create_dataset.py → data.pickle).
3. Train a classifier (train_classifier.py) and save the model (model.p).
4. Run real-time inference (inference_classifier.py).
   ## Demo video
   Watch how the final output turns out :
   https://drive.google.com/file/d/19o4bFDA4UDYIvdX1haQ6sx7ArQoNlJhv/view

## Features
- Webcam-based data capture (class folders).
- Landmark extraction using MediaPipe Hands.
- Normalized feature vectors for stable classification.
- RandomForest classifier for sign recognition.
- Real-time prediction and visualization (bounding box + predicted label).

## Technologies / Tools used
- Python 3.8+  
- OpenCV (opencv-python)  
- MediaPipe (mediapipe==0.10.1)  
- NumPy, Matplotlib  
- scikit-learn (RandomForest)  
- Git + GitHub

## Steps to install & run the project

### Setup (Windows PowerShell)
```powershell
# create and activate venv (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# install dependencies
pip install --upgrade pip
pip install -r requirements.txt
