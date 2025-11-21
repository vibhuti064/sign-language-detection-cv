
# Sign Language Detection (CV Project)
![Inference Demo](https://github.com/vibhuti064/sign-language-detection-cv/blob/main/ASL_Alphabet.jpg)

## Project title
*Sign Language Detection (Computer Vision)*

## Overview of the project
![Demo](https://github.com/vibhuti064/sign-language-detection-cv/blob/main/ScreenRecording2025-11-21233717-ezgif.com-crop.gif)









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
### Prerequisites

1.  Ensure you have **Python 3.9 - 3.12** installed.
2.  Your project folder must contain the following files: `collect_imgs.py`, `create_dataset.py`, `train_classifier.py`, `inference_classifier.py`, and the entire `data` directory with your images.

### Installation

1.  **Activate Virtual Environment:** Open your terminal and activate your virtual environment.
    ```bash
    .\venv\Scripts\activate
    ```
2.  **Install Dependencies:** Install the required Python libraries.
    ```bash
    pip install mediapipe opencv-python scikit-learn numpy
    ```

### Running the Classifier

1.  **Train the Model:** Run the training script (this will generate the `model.p` file).
    ```bash
    python .vscode\train_classifier.py
    ```
2.  **Start Real-Time Inference:** Run the final script to open the webcam and start detection.
    ```bash
    python .vscode\inference_classifier.py
    ```

## ✅ Instructions for Testing

1.  Run the inference script as described above.
2.  Hold your hand steady, ensuring it is well-lit and clearly visible to the camera.
3.  Perform the 10 trained ISL gestures one by one. The corresponding letter should be displayed in the bounding box around your hand.
4.  To close the camera feed and stop the program, click on the camera window and press the **'q' key**.
