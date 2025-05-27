# Emotion By AI

## About
**Emotion By AI** lets users upload a photo and instantly detect the emotion expressed by the person in it using deep learning models and facial landmark features.

---

## Description

### Dataset
- Downloaded train and test datasets from **Kaggle**.

### Model Training
Trained the dataset using the following models:

1. **Custom Model**:
   - Built a Convolutional Neural Network from scratch using **PyTorch**.

2. **Pretrained Models**:
   - ResNet-18  
   - ResNet-34  
   - ResNet-50  
   - VGG-16  

3. **MediaPipe-Based Approach**:
   - Used **MediaPipe Face Mesh** to extract 3D facial landmarks for emotion classification.

### Model Evaluation
- Validated all models using test data and evaluated them using **Accuracy**, **Precision**, **Recall**, and **F1-score**.

### Streamlit Interface
- Built a user-friendly **Streamlit** interface that allows users to upload an image.
- Automatically detects and classifies the emotion in the image and displays the **predicted emotion with a confidence score**.

---

## Installation

Environment: Google Colab

Install the required libraries:
```bash
!pip install torch
!pip install torchvision
!pip install numpy
!pip install sklearn
!pip install pandas
!pip install mediapipe
!pip install streamlit
!pip install pyngrok   # Only if using Colab
!pip install opencv-python

## Importing

- `numpy`: Used for data typecasting and selecting random images in the dataset based on specific conditions.
- `torch`: Used for building and training neural networks.
- `torch.nn`, `torch.optim`: For defining neural network layers and optimization algorithms.
- `torchvision`: Provides utilities for image transformation and access to pretrained models.
- `torchvision.transforms`, `torchvision.models`: Used for data augmentation and loading models like ResNet and VGG.
- `torch.utils.data`: Helps in batching the data and custom dataset handling.
- `sklearn.metrics` (`accuracy_score`, `precision_score`, `f1_score`, `recall_score`): Used to evaluate model performance.
- `os`: Used for file path manipulations and directory creation.
- `cv2` (OpenCV): Used for reading and processing images directly from directories.
- `streamlit`: Used for building an interactive user interface for image upload and emotion prediction.
- `PIL.Image`: Used for loading and handling images, especially in conjunction with torchvision.
- `pyngrok`: Used (only in Google Colab) to create public URLs for accessing the Streamlit app from external devices.

## Selection of Appropriate Model

### ✅ Pretrained Models Workflow

- **Data Augmentation:**  
  Applied using `torchvision.transforms`:
  - Convert images to 3-channel **RGB**
  - **RandomHorizontalFlip**
  - **RandomRotation**
  - **ColorJitter**
  - **RandomResizedCrop** to (224, 224)
  - Normalize using ImageNet mean and std

- **Class Balancing:**  
  Selected the class with the fewest samples and uniformly sampled the same number of images from other classes.

- **Model Fine-tuning Steps:**
  - **Freeze initial layers** of pretrained models
  - Train only the **final layers**
  - Replace the final classification layer to match the number of emotion classes
  - Add **ReLU** and **Dropout** layers to reduce overfitting
  - Use **Adam optimizer** with `lr=0.0001` and `weight_decay=1e-4`
  - Use **CrossEntropyLoss** as the loss function
  - Apply a **learning rate scheduler** to improve convergence

---

### 🔹 ResNet-18

**Training Performance:**
- Accuracy: `0.6501`
- Precision: `0.6501`
- Recall: `0.6501`
- F1-score: `0.6501`

**Testing Performance:**
- Accuracy: `0.5122`
- Precision: `0.5122`
- Recall: `0.5122`
- F1-score: `0.5122`

➡️ *Overfitting observed*

---

### 🔹 ResNet-34

**Training Performance:**
- Accuracy: `0.6364`
- Precision: `0.6364`
- Recall: `0.6364`
- F1-score: `0.6364`

**Testing Performance:**
- Accuracy: `0.5187`
- Precision: `0.5187`
- Recall: `0.5187`
- F1-score: `0.5187`

➡️ *Overfitting observed*

---

### 🔹 VGG-16

**Training Performance:**
- Accuracy: `0.7207`
- Precision: `0.7207`
- Recall: `0.7207`
- F1-score: `0.7207`

**Testing Performance:**
- Accuracy: `0.5431`
- Precision: `0.5431`
- Recall: `0.5431`
- F1-score: `0.5431`

➡️ *Overfitting observed*

---

### 🔹 ResNet-50

#### 1. Down Sampling
- Sampled same number of images from each class based on the smallest class.
- **Training Accuracy:** `0.7120`
- Precision, Recall, F1-score: `0.7120`

#### 2. Over Sampling
- Oversampled all classes to match the largest class size.
- **Training Accuracy:** `0.5707`
- Precision: `0.5496`
- Recall: `0.5707`
- F1-score: `0.5519`

#### 3. Class Weight Balancing
- Used `WeightedRandomSampler` with inverse frequency weights.
- **Training Accuracy:** `0.7420`
- Precision: `0.7398`
- Recall: `0.7419`
- F1-score: `0.7401`

**Test Performance:**
- Accuracy: `0.6705`
- Precision: `0.6774`
- Recall: `0.6705`
- F1-score: `0.6696`

✅ *Weighted sampling gave the best results with ResNet-50.*

---

## 📌 MediaPipe-Based Emotion Classifier

- **Extracted 3D facial landmarks** (468 points) using **MediaPipe Face Mesh**
- Saved features per image into CSV files, labeled with the corresponding class (`emotion`)
- Combined all class-wise CSVs into a **single DataFrame**
- Split data into **features (X)** and **labels (y)**
- **Outlier detection** performed (and removed if needed)
- Applied **StandardScaler** and saved using `pickle`
- Reduced features to **256 components** using **PCA**, saved as well
- **Label encoded** the target values and saved the encoder
- Converted features and labels to **PyTorch tensors**
- Designed a **5-layer Sequential Model** with:
  - Linear → BatchNorm → ReLU → Dropout → Output
- Trained using:
  - **Adam optimizer**
  - **CrossEntropyLoss**
  - **Scheduler**
  - **DataLoader**

**Training Performance:**
- Accuracy: `0.5716`
- Precision: `0.5112`
- Recall: `0.5953`
- F1-score: `0.5276`

**Testing Performance:**
- Accuracy: `0.5527`
- Precision: `0.4790`
- Recall: `0.5463`
- F1-score: `0.4904`

---

## ✅ Final Model Selection

After evaluating all models, **ResNet-50 with class weight balancing using WeightedRandomSampler** provided the most consistent and highest performance for image-based emotion classification. Therefore, it is selected as the **best-performing model** for deployment.

## 📲 Streamlit User Interface

- Developed an interactive **Streamlit** interface that allows users to **upload an image** directly through the web application.
- Once an image is uploaded, the system:
  - Processes the image
  - Uses the **trained model** to predict the emotion
  - Displays the **predicted emotion** as the final output on the screen

---

## ✅ Conclusion

This project explored emotion classification using **custom-built CNNs**, **pretrained deep learning models**, and **MediaPipe-based 3D facial landmark extraction**.  
Among all approaches, **ResNet-50** combined with **weighted random sampling** provided the highest accuracy and generalization performance.  
A user-friendly **Streamlit interface** was also developed, making the solution interactive and accessible.  
Overall, the system offers an effective and intuitive method for **real-time emotion detection from images**.
