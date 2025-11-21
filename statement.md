Project Statement

Problem statement

Many people rely on sign language to communicate. The goal of this project is to build a lightweight hand sign detection system that recognizes a fixed set of hand signs (A–J) in real-time using a webcam. The system uses hand landmark features to make classification robust to lighting and background variations.

Scope of the project

Capture images of hand signs using the webcam.

Use MediaPipe to extract 21 hand landmarks per hand.

Build a normalized landmark-based dataset for training.

Train a RandomForest classifier and evaluate performance.

Deploy a real-time inference script to detect signs via webcam.


This project focuses on a limited vocabulary (10 signs). Extending to more signs or continuous sign language recognition is future work.

Target users

Students learning computer vision and human-computer interaction.

Developers building accessibility tools (proof-of-concept).

Researchers wanting a small, reproducible dataset and pipeline for hand pose classification.


High-level features

Webcam-based data capture with organized class folders.

Landmark normalization to create consistent features.

Model training script with test/train split and accuracy reporting.

Real-time inference with landmark visualization and predicted label.
