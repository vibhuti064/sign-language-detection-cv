## Project Statement
SIGN LANGUAGE DETECTION WITHIN THE FIED OF COMPUTER VISION

## Problem statement

Many people rely on sign language to communicate. The goal of this project is to build a lightweight hand sign detection system that recognizes a fixed set of hand signs (A–J) in real-time using a webcam. The system uses hand landmark features to make classification robust to lighting and background variations.

## 🔭 Scope of the Project

The scope is limited to a **static, single-hand ISL alphabet recognition system**.

* **Input:** Live video stream from a standard webcam.
* **Output:** Predicted ISL character displayed on the screen.
* **Environment:** Windows 11 desktop environment using Python and common ML libraries.
* **Data Set:** 10 unique ISL gestures/letters.

## 👤 Target Users

* **Hearing-impaired individuals** and their families seeking accessible communication tools.
* **Developers** interested in integrating gesture-based controls into applications.
* **Educators** creating interactive learning tools for sign language.
* **Researchers** needing a reliable computer vision baseline for gesture classification.

## 🌟 High-Level Features

1.  **Landmark Feature Extraction:** Use MediaPipe to consistently extract 42 2D hand landmark coordinates (normalized to 0-1).
2.  **Feature Normalization:** Implement custom logic to make the features **translation-invariant** (independent of the hand's position on screen).
3.  **Model Training:** Train a classification model (Random Forest) capable of distinguishing the 10 classes with high accuracy.
4.  **Real-time Prediction:** Integrate the trained model with OpenCV for live, low-latency gesture prediction and visualization.
