# Automated Fabric Defect Classification for Textile Quality Inspection Using Deep Learning

## 📌 Overview

This project develops a deep learning-based computer vision system for automatically classifying defects in fabric images. The goal is to support quality inspection in textile manufacturing by identifying different types of fabric defects quickly and accurately.

The system uses transfer learning with pretrained convolutional neural networks (CNNs) to classify fabric images into multiple defect categories.

---

## 🎯 Objectives

- Develop a multi-class image classification model for fabric defect recognition.
- Explore and preprocess a public fabric defect dataset.
- Train and evaluate deep learning models using transfer learning.
- Compare model performance using standard evaluation metrics.
- Build a reproducible machine learning pipeline.

---

## 🏭 Problem Statement

Quality inspection in textile manufacturing is traditionally performed manually. This process is repetitive, time-consuming, and may lead to inconsistent results due to human fatigue.

This project aims to automate the classification of fabric defects using computer vision and deep learning, helping inspectors identify defective fabrics more efficiently.

---

## 🧵 Dataset

**Dataset:** Multi-Class Fabric Defect Detection Dataset

The dataset contains over 3,000 high-resolution fabric images collected from real textile production lines. Images belong to nine classes including:

- Defect Free
- Hole
- Horizontal
- Vertical
- Lines
- Pinched Fabric
- Needle Mark
- Broken Stitch
- Stain

**Sample images per class:**

![Sample fabric defect images per class](assets/sample_defects_grid.png)

**Class distribution:**

![Fabric defect class distribution](assets/class_distribution.png)

The dataset is imbalanced — "Defect Free" accounts for ~54% of images, while classes like "Vertical" make up only ~3%. This is addressed during model training via class weighting/augmentation (see `notebooks/01_dataset_exploration.ipynb` for the full analysis).

---

## 🛠️ Technologies

- Python
- PyTorch
- OpenCV
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Google Colab
- Git
- GitHub

---

## 🧠 Machine Learning Approach

The project follows a transfer learning approach using pretrained convolutional neural networks such as:

- ResNet18
- EfficientNet-B0

The workflow includes:

1. Data exploration
2. Data preprocessing
3. Data augmentation
4. Model training
5. Model evaluation
6. Prediction on unseen images

---

## 📊 Evaluation Metrics

Model performance will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 📁 Project Structure

```
Fabric-Defect-Classification/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
├── models/
├── outputs/
├── docs/
└── assets/
```

---

## 🚀 Getting Started

Clone the repository:

```bash
git clone https://github.com/yourusername/Fabric-Defect-Classification.git
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## 📈 Future Improvements

- Real-time fabric inspection using industrial cameras.
- Fabric defect localisation using object detection.
- Edge deployment for embedded devices.
- Defect severity estimation.
- Integration into automated textile production systems.

---

## 📚 Learning Outcomes

Through this project, I aim to strengthen my knowledge of:

- Computer Vision
- Deep Learning
- Transfer Learning
- Image Classification
- Model Evaluation
- Machine Learning Engineering
- Git and GitHub

---

## 📄 License

This project was developed for educational purposes as part of an AI/ML capstone project.