# Environmental-Sound-Classification-using-CNN
A deep learning-based Environmental Sound Classification (ESC) system that uses Mel Frequency Cepstral Coefficients (MFCCs) and a Convolutional Neural Network (CNN) to classify environmental audio recordings into five sound categories.
📌 Project Overview

Environmental Sound Classification enables machines to automatically recognize sounds from their surroundings.

In this project, audio recordings are transformed into MFCC feature representations and supplied to a CNN model for classification.

The system was developed using the UrbanSound8K dataset and focuses on five environmental sound categories:

🐕 Dog Bark
🚗 Car Horn
❄️ Air Conditioner
🚨 Siren
🚘 Engine Idling

The trained model achieved a 92.32% accuracy on the held-out test set.

The project also includes an inference pipeline that accepts previously unseen external .wav recordings and produces a predicted class along with class-wise confidence scores.

🎯 Objectives

The main objectives of this project are:

Load and explore an environmental audio dataset.
Select relevant environmental sound classes.
Extract meaningful acoustic features using MFCC.
Convert audio features into a CNN-compatible representation.
Develop and train a CNN-based classifier.
Evaluate the model using standard classification metrics.
Test the trained model on unseen audio recordings.
Analyze prediction confidence and model behavior.
🗂️ Dataset

This project uses the UrbanSound8K environmental sound dataset.

UrbanSound8K contains labeled environmental audio recordings distributed across ten predefined sound classes and ten folds.

Selected Classes
Class	Description
dog_bark	Dog barking sounds
car_horn	Vehicle horn sounds
air_conditioner	Air-conditioning sounds
siren	Emergency vehicle siren sounds
engine_idling	Vehicle engine idling sounds
Dataset Source

UrbanSound8K:

https://urbansounddataset.weebly.com/urbansound8k.html

Note: The dataset itself is not included in this repository because of its size and dataset distribution considerations. Download it separately and place it in the expected directory structure when running the notebook.

🧠 Methodology

The complete workflow is:

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
                 MFCC Extraction
                  (40 Coefficients)
                         │
                         ▼
              Padding / Truncation
                         │
                         ▼
                 Label Encoding
                         │
                         ▼
              Train / Test Split
                    (80 / 20)
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
🔬 Feature Extraction
Why MFCC?

Raw audio signals contain a large number of amplitude samples and are not directly efficient for the CNN to process.

Mel Frequency Cepstral Coefficients (MFCCs) provide a compact representation of the spectral characteristics of an audio signal.

MFCCs are useful because they:

Capture important frequency characteristics.
Reduce the dimensionality of the raw audio representation.
Approximate aspects of human auditory perception.
Are widely used in audio and speech classification applications.
MFCC Configuration
Number of MFCC coefficients : 40
Fixed time dimension        : 174 frames
Input representation        : 40 × 174
CNN channels                : 1

Audio recordings with fewer than 174 frames are padded, while longer recordings are truncated to maintain a fixed input size.

🧹 Data Preprocessing

The following preprocessing pipeline is applied:

WAV Audio
    ↓
Audio Loading
    ↓
MFCC Extraction
    ↓
Padding / Truncation
    ↓
NumPy Array
    ↓
Label Encoding
    ↓
One-Hot Encoding
    ↓
CNN Channel Dimension
    ↓
Feature Normalization
    ↓
Train/Test Split
Dataset Split
Training Set : 80%
Testing Set  : 20%

A stratified split is used to preserve the distribution of the selected sound classes.

🤖 CNN Architecture

The project uses a convolutional neural network designed to learn patterns from the two-dimensional MFCC representation.

Architecture
Input
40 × 174 × 1
      │
      ▼
Conv2D
32 Filters
3 × 3 Kernel
ReLU
      │
      ▼
Batch Normalization
      │
      ▼
MaxPooling
      │
      ▼
Dropout
30%
      │
      ▼
Conv2D
64 Filters
3 × 3 Kernel
ReLU
      │
      ▼
Batch Normalization
      │
      ▼
MaxPooling
      │
      ▼
Dropout
30%
      │
      ▼
Flatten
      │
      ▼
Dense
128 Neurons
ReLU
      │
      ▼
Dropout
50%
      │
      ▼
Softmax
5 Classes
⚙️ Training Configuration
Parameter	Value
Optimizer	Adam
Loss Function	Categorical Crossentropy
Batch Size	32
Maximum Epochs	30
Activation	ReLU
Output Activation	Softmax
EarlyStopping	Enabled
ModelCheckpoint	Enabled
Input Size	40 × 174 × 1
EarlyStopping

EarlyStopping is used to stop training when validation performance stops improving, helping reduce unnecessary training and overfitting.

ModelCheckpoint

ModelCheckpoint saves the best-performing model based on validation accuracy.

The best model is saved as:

Models/best_model.keras
📊 Results

The model was evaluated on the held-out test set that was not used for model parameter learning.

Final Test Performance
Metric	Result
Test Accuracy	92.32%
Precision	~93%
Recall	~92%
F1-Score	~92%

The exact per-class precision, recall and F1 values are available in the classification report generated by the notebook.

📈 Training Performance

The notebook generates:

Training Accuracy vs Validation Accuracy
Training Loss vs Validation Loss
Confusion Matrix
Classification Report
Prediction Probability Distribution

Place the generated figures in the Results/ directory.

Example:

Results/
├── class_distribution.png
├── waveform.png
├── spectrogram.png
├── mfcc.png
├── accuracy_curve.png
├── loss_curve.png
├── confusion_matrix.png
└── workflow_diagram.png
🧪 Testing on Unseen Audio

After training, the model can be tested with a new .wav recording that was not part of the training process.

The inference pipeline is:

External WAV File
       ↓
Audio Loading
       ↓
MFCC Extraction
       ↓
Padding / Truncation
       ↓
Normalization
       ↓
CNN
       ↓
Softmax Probabilities
       ↓
Predicted Class
       ↓
Confidence Score

The system displays the probability assigned to each of the five classes.

Example:

Prediction Probabilities

car_horn        : 52.75%
air_conditioner : 24.55%
engine_idling   : 17.48%
dog_bark        :  3.44%
siren           :  1.78%

Final Prediction
Class      : car_horn
Confidence : 52.75%
Important Note

Confidence on external audio should not be interpreted as overall model accuracy.

The 92.32% accuracy represents performance on the held-out test set, whereas an external recording tests how well the model generalizes to audio outside the original dataset distribution.

External recordings can differ in:

Microphone characteristics
Background noise
Recording environment
Audio quality
Sound intensity
Acoustic characteristics

Therefore, lower confidence on an external recording does not directly mean that the model's test accuracy is low.

📁 Repository Structure
Environmental-Sound-Classification-CNN/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── Notebook/
│   └── Environmental_Sound_Classification_CNN.ipynb
│
├── Dataset/
│   └── dataset_link.txt
│
├── Models/
│   ├── best_model.keras
│   └── final_model.keras
│
├── Results/
│   ├── class_distribution.png
│   ├── waveform.png
│   ├── spectrogram.png
│   ├── mfcc.png
│   ├── model_summary.png
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   └── workflow_diagram.png
│
└── Report/
    └── Environmental_Sound_Classification_Report.pdf
💻 Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/Environmental-Sound-Classification-CNN.git

Move into the project directory:

cd Environmental-Sound-Classification-CNN

Install the required Python packages:

pip install -r requirements.txt
🚀 Running the Project
Option 1 — Google Colab

The project was developed and trained using Google Colab.

Open the notebook:
Notebook/Environmental_Sound_Classification_CNN.ipynb
Open it using Google Colab.
Mount Google Drive.
Download the UrbanSound8K dataset.
Place the dataset in the expected directory.
Run the notebook cells sequentially.
Option 2 — Local Environment

Install the dependencies:

pip install -r requirements.txt

Download the UrbanSound8K dataset and configure the dataset path in the notebook.

Then run:

Environmental_Sound_Classification_CNN.ipynb

using Jupyter Notebook or JupyterLab.

🛠️ Technologies Used
Technology	Purpose
Python	Programming
TensorFlow	Deep Learning
Keras	CNN Development
Librosa	Audio Processing
NumPy	Numerical Computation
Pandas	Dataset / Metadata Processing
Scikit-learn	Data Splitting & Evaluation
Matplotlib	Visualization
Google Colab	Development & Training
Google Drive	Dataset / Model Storage
📌 Key Implementation Decisions
Why CNN?

MFCCs form a two-dimensional time-frequency representation. CNNs can learn local patterns from this representation and automatically identify discriminative acoustic features.

Why MFCC?

MFCC provides a compact representation of the spectral properties of audio and is commonly used for audio classification.

Why Softmax?

The problem is a multi-class classification problem involving five mutually exclusive classes. Softmax converts the model's final outputs into class probabilities.

Why Dropout?

Dropout reduces over-reliance on specific neurons and helps reduce overfitting.

Why Batch Normalization?

Batch normalization helps stabilize the training process and can improve convergence.

Why Adam?

Adam provides adaptive learning rates for model parameters and is effective for training deep neural networks.

⚠️ Limitations

The current system has several limitations:

The classifier recognizes only five selected environmental sound categories.
External recordings can have different acoustic characteristics from the training dataset.
The system is currently designed for .wav audio input.
Real-time microphone classification has not been implemented in the current version.
The model has not yet been converted to TensorFlow Lite for edge deployment.
The system may produce low-confidence predictions for sounds that do not belong to the five trained classes.
🔮 Future Improvements

Potential future improvements include:

Data augmentation using noise addition, time shifting and pitch variation.
Improved robustness against background noise.
Real-time microphone-based classification.
TensorFlow Lite conversion.
Deployment on edge devices.
Expansion to additional environmental sound categories.
Evaluation using larger and more diverse external datasets.
Confidence-based unknown-class detection.
📄 Project Report

A detailed technical report describing the methodology, feature extraction, CNN architecture, training process, evaluation and results is available in:

Report/Environmental_Sound_Classification_Report.pdf
👨‍💻 Author

Sumith Rwddy

Electronics and Computer Engineering

📜 License

This project is released under the MIT License.

See LICENSE for details.

⭐ Acknowledgements

This project uses the UrbanSound8K dataset for environmental sound classification research and experimentation.

Dataset:

https://urbansounddataset.weebly.com/urbansound8k.html

📌 Project Status

Completed

The current implementation includes:

Dataset preparation

Exploratory data analysis

MFCC feature extraction

Data preprocessing

CNN model development

Model training

Early stopping

Best model checkpointing

Test-set evaluation

Classification report

Confusion matrix

External unseen audio testing

Prediction confidence analysis

Technical report

⭐ Final Result

92.32% Test Accuracy on the held-out test set

The project demonstrates an end-to-end deep learning pipeline for environmental sound classification, from raw audio preprocessing and MFCC extraction to CNN-based classification and evaluation on unseen data.
