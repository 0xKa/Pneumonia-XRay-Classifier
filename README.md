# Pneumonia X-Ray Classifier (Binary Image Classification)

This project uses **deep learning** techniques to classify chest X-ray images into two categories: **Pneumonia** and **Normal**. It applies both a **Custom Convolutional Neural Network (CNN)** and a **Transfer Learning approach (MobileNetV2)** to compare performance on a real-world medical diagnosis task.

> ⚠️ Dataset not included in this repo due to large size — follow the instructions below to download and set it up locally.

---

## 📁 Project Structure

pneumonia-xray-classifier/
├── dataset/
│   ├── train/
│   │   └── >> Contains Images of NORMAL/ and PNEUMONIA/
│   └── test/
│       └── >> Contains Images of NORMAL/ and PNEUMONIA/
├── models/
│   └── >> Saved trained models (.keras)
├── nootbooks/
│   ├── 1_data_exploration.ipynb
│   ├── 2_train_custom_cnn.ipynb
│   ├── 3_train_transfer_model.ipynb
│   └── 4_evaluate_model.ipynb
├── .gitignore
├── README.md
└── requirements.txt

---

## 🧪 Dataset

- **Name:** Chest X-Ray Images (Pneumonia)
- **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Size:** ~2 GB
- **Classes:** `NORMAL`, `PNEUMONIA`

### 🔽 Download Instructions

1. Visit the [dataset page on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Download and extract it
3. Place the dataset folder like this: \
   dataset/\
   └── train/\
   └── test/

---

## ⚙️ Setup Instructions

1.  Clone this repo:
    ```bash
    git clone https://github.com/your-username/pneumonia-xray-classifier.git
    cd pneumonia-xray-classifier
    ```
2.  Create a virtual environment:
    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    # or source .venv/bin/activate on Linux/MacOS
    ```
3.  Install dependencies:

    ```
    pip install -r requirements.txt
    ```

4.  Open notebooks:
    ```
    jupyter notebook
    ```

---

## 🧠 Models Used

### ✅ Custom CNN

- Built from scratch using Keras Sequential API
- Architecture includes:
  - 3 convolutional layers with ReLU activation
  - MaxPooling layers
  - Fully connected Dense layers
  - Dropout to reduce overfitting
- Output layer with sigmoid activation for binary classification

### ✅ MobileNetV2 (Transfer Learning)

- Pre-trained on ImageNet dataset
- Only the top layers (classification head) are trainable
- Fast convergence with better generalization
- Architecture:
  - GlobalAveragePooling
  - Dense layers with dropout
  - Sigmoid output layer

---

## 📈 Results (Sample)

| Model       | Accuracy | Precision | Recall | F1 Score |
| ----------- | -------- | --------- | ------ | -------- |
| Custom CNN  | 0.76     | 0.72      | 0.99   | 0.836    |
| MobileNetV2 | 0.83     | 0.79      | 0.99   | 0.880    |

> 🔬 These results are based on evaluation using the same test set. Actual values may vary depending on training epochs, hardware, and random initialization.

---

## 📄 License & Credits

- Dataset by Paul Mooney via [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Transfer learning model: [MobileNetV2](https://arxiv.org/abs/1801.04381) from TensorFlow/Keras
- Project developed for my AI&DL Assignment

---
