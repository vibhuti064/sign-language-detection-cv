
# 🖐️ Sign Language Detection (Computer Vision Project)

![Inference Demo](https://github.com/vibhuti064/sign-language-detection-cv/blob/main/ASL_Alphabet.jpg)

---

## 📌 Project Overview

![Demo](https://github.com/vibhuti064/sign-language-detection-cv/blob/main/ScreenRecording2025-11-21233717-ezgif.com-crop.gif)

This project implements a **real-time sign language recognition system** using **Computer Vision and Machine Learning**. It captures hand gestures through a webcam, processes them using **MediaPipe hand landmarks**, and predicts the corresponding sign using a trained ML model.

The system is designed as an **end-to-end pipeline**, covering data collection, feature engineering, model training, and real-time inference.

---

## 🧠 How It Works (Pipeline)


## Data Collection → Landmark Extraction → Feature Engineering → Model Training → Real-time Prediction
🔹 Step 1: Data Collection

Capture gesture images using webcam
## 📹 Demo:
https://drive.google.com/file/d/1Q4d-tTJtkU5jnwB-6m7SN8JmQUEBnS0w/view?usp=sharing

🔹 Step 2: Dataset Creation

Convert images into structured landmark-based dataset (data.pickle)

🔹 Step 3: Model Training

Train ML models (Random Forest, SVM, KNN) and select best model

🔹 Step 4: Real-time Inference

Run live prediction with webcam
## 📹 Demo:
https://drive.google.com/file/d/19o4bFDA4UDYIvdX1haQ6sx7ArQoNlJhv/view

## 🚀 Key Features
📷 Real-time webcam-based gesture detection
✋ Hand landmark extraction using MediaPipe (21 landmarks → 42 features)
⚙️ Feature normalization for stable predictions
🤖 Machine Learning models (Random Forest, SVM, KNN)
📊 Model evaluation using confusion matrix and classification metrics
🎯 Real-time bounding box visualization with predicted labels

## 🛠️ Tech Stack
Python 3.8+
OpenCV – Image processing & webcam handling
MediaPipe – Hand landmark detection
Scikit-learn – ML models & evaluation
NumPy & Matplotlib – Data handling & visualization
Git & GitHub – Version control

## 📊 Model Performance

All models achieved near-perfect accuracy on the dataset:

Random Forest: 100%
SVM: 100%
KNN: 100%

💡 High accuracy is achieved due to structured landmark-based features and controlled data collection environment, making gesture classes highly separable.

## 📂 Project Structure
.
├── collect_imgs.py          
├── create_dataset.py        
├── train_classifier.py      
├── inference_classifier.py 

├── requirements.txt

└── README.md

## ⚙️ Setup & Installation
🔹 Prerequisites
Python 3.9 – 3.12
Webcam access
🔹 Installation
.\venv\Scripts\activate
pip install mediapipe opencv-python scikit-learn numpy matplotlib

## ▶️ Run the Project
1️⃣ Train the Model
python train_classifier.py

2️⃣ Run Real-time Detection
python inference_classifier.py

## 🧪 Testing Instructions
Run the inference script
Ensure proper lighting and clear hand visibility
Perform gestures (A–J)
Observe predicted label in real-time
Press 'q' to exit

## 🎯 Key Learnings
Feature engineering using landmark normalization
Building end-to-end ML pipelines
Model comparison and evaluation techniques
Real-time deployment of ML models
Integration of Computer Vision with Machine Learning

## ⚠️ Limitations
Works best in controlled environments
Limited gesture set (A–J)
Performance may vary with lighting/background

## 🚀 Future Improvements
Add Deep Learning models (CNN / LSTM)
Expand dataset with more gestures
Improve robustness for real-world scenarios
Deploy as web/mobile application

👤 Author
Vibhuti Purohit

⭐ If you found this project useful, consider giving it a star!


