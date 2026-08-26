# 🎧 Environmental Sound Classification using CNN

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras)
![Librosa](https://img.shields.io/badge/Librosa-Audio%20Processing-green)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebook-yellow?logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-green)

A deep learning-based **Environmental Sound Classification (ESC)** system using **Mel Frequency Cepstral Coefficients (MFCCs)** and a **Convolutional Neural Network (CNN)** to classify environmental audio recordings into five sound categories.

---

## **📌 Project Overview**

Environmental Sound Classification enables machines to automatically recognize sounds from their surroundings.

This project transforms environmental audio recordings into MFCC feature representations and uses a CNN model to classify them into predefined sound categories.

The project uses the **UrbanSound8K** dataset and focuses on five environmental sound classes:

- 🐕 **Dog Bark**
- 🚗 **Car Horn**
- ❄️ **Air Conditioner**
- 🚨 **Siren**
- 🚘 **Engine Idling**

The trained model achieved a **92.32% accuracy on the held-out test set**.

The project also contains an inference pipeline for testing previously unseen `.wav` audio recordings and generating class-wise prediction probabilities and confidence scores.

---

## **🎯 Objectives**

The main objectives of this project are:

1. Load and explore an environmental audio dataset.
2. Select relevant environmental sound classes.
3. Extract meaningful acoustic features using MFCC.
4. Convert audio features into a CNN-compatible representation.
5. Develop and train a CNN-based classifier.
6. Evaluate the trained model using standard classification metrics.
7. Test the trained model on unseen audio recordings.
8. Analyze prediction probabilities and confidence scores.

---

## **🗂️ Dataset**

This project uses the **UrbanSound8K** environmental sound dataset.

UrbanSound8K is a benchmark dataset containing labeled environmental audio recordings distributed across multiple environmental sound categories.

### **Selected Classes**

| Class | Description |
|---|---|
| `dog_bark` | Dog barking sounds |
| `car_horn` | Vehicle horn sounds |
| `air_conditioner` | Air-conditioning sounds |
| `siren` | Emergency vehicle siren sounds |
| `engine_idling` | Vehicle engine idling sounds |

### **Dataset Source**

UrbanSound8K:

https://urbansounddataset.weebly.com/urbansound8k.html

> **Note:** The dataset is not included in this repository because of its size and distribution considerations. Download the dataset separately before running the notebook.

---

## **🔬 Methodology**

The complete project workflow is:

```text
                 UrbanSound8K Dataset
                         │
                         ▼
                  Load Metadata
                         │
                         ▼
                 Select 5 Classes
                         │
                         ▼
                  Load WAV Files
                         │
                         ▼
              Exploratory Data Analysis
              ┌──────────┴──────────┐
              │                     │
           Waveform             Spectrogram
              └──────────┬──────────┘
                         ▼
              MFCC Feature Extraction
                  (40 Coefficients)
                         │
                         ▼
                Padding / Truncation
                         │
                         ▼
                   Label Encoding
                         │
                         ▼
                  Train/Test Split
                       (80/20)
                         │
                         ▼
                 Feature Normalization
                         │
                         ▼
                     CNN Model
                         │
                         ▼
                      Training
              ┌──────────┴──────────┐
              │                     │
        EarlyStopping       ModelCheckpoint
              └──────────┬──────────┘
                         ▼
                   Model Evaluation
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       Accuracy     Classification   Confusion
                      Metrics          Matrix
                         │
                         ▼
                Unseen Audio Prediction
                         │
                         ▼
              Class + Confidence Score




