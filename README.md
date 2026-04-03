# 🌿 Plant Disease Classifier

A Convolutional Neural Network (CNN) built **from scratch** to classify plant leaf diseases across three categories — trained on image data and achieving multi-class prediction using TensorFlow/Keras.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bishowAd/plant-disease-classifier/blob/main/Plant.ipynb)

---

##  Overview

This project uses deep learning to identify plant diseases from leaf images. Given a photo of a plant leaf, the model predicts whether it belongs to one of three disease categories:

- **Corn Rust**
- **Potato Early Blight**
- **Tomato Bacterial Spot**

The CNN is constructed entirely from scratch using Keras — no transfer learning or pre-trained models — making this a strong demonstration of foundational deep learning principles.

---

## Model Architecture

| Layer | Details |
|---|---|
| Conv2D | 32 filters, 3×3 kernel, ReLU, same padding |
| MaxPooling2D | 3×3 pool size |
| Conv2D | 64 filters, 3×3 kernel, ReLU, same padding |
| MaxPooling2D | 2×2 pool size |
| Flatten | — |
| Dense | 64 units, ReLU |
| Dense (Output) | 3 units, Softmax |

- **Loss Function:** Categorical Cross-Entropy  
- **Optimizer:** Adam  
- **Input Shape:** 256 × 256 × 3 (RGB images)  
- **Output:** 3-class softmax probability vector  

---

## Dataset

Images are organized into three class folders stored in Google Drive:

```
Plant_image/
├── Corn_Rust/
├── Potato___Early_blight/
└── Tomato_Bacterial_Spot/
```

Each image is:
- Read using OpenCV (`cv2`)
- Resized to **256 × 256**
- Converted to a NumPy array
- Normalized to the `[0, 1]` range

Labels are one-hot encoded using `to_categorical`.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| TensorFlow / Keras | Model building and training |
| OpenCV (`cv2`) | Image reading and resizing |
| NumPy / Pandas | Array ops and label counting |
| Matplotlib | Training visualization |
| Scikit-learn | Train/test/validation splitting |
| Google Colab | Training environment |

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/bishowAd/plant-disease-classifier.git
cd plant-disease-classifier
```

### 2. Install dependencies

```bash
pip install tensorflow opencv-python pillow scikit-learn matplotlib pandas numpy
```

### 3. Prepare your dataset

Organize your images in the folder structure shown above and update the `dir` path in the notebook to point to your dataset location.

### 4. Run the notebook

Open `Plant.ipynb` in Jupyter or Google Colab and run all cells. The notebook walks through:

1. Mounting Google Drive and exploring the dataset
2. Converting images to NumPy arrays
3. Splitting into train / validation / test sets
4. Building and compiling the CNN
5. Training for 10 epochs with batch size 128
6. Plotting training vs. validation accuracy
7. Evaluating on the test set
8. Making and visualizing individual predictions

---

Training

```
Epochs:     10
Batch Size: 128
Val Split:  20% of training data
Test Split: 20% of full dataset
```

Training and validation accuracy are plotted after training to visualize model convergence.

---

Sample Prediction

```python
print("Original  :", all_labels[np.argmax(y_test[10])])
print("Predicted :", all_labels[np.argmax(y_pred[10])])
```

```
Original  : Potato___Early_blight
Predicted : Potato___Early_blight
```

---

##  Saving the Model

The trained model is saved in Keras format for future inference:

```python
model.save('/content/drive/My Drive/Plant_image/model.keras')
```

---

##  Project Structure

```
plant-disease-classifier/
├── Plant.ipynb       # Main notebook
└── README.md         # You are here
```

---
