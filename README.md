# Capstone-project
# 🚗 RoadGuardian AI

YOLO-Based Road Damage Detection Using Computer Vision

---

## 📌 Overview



RoadGuardian AI is a computer vision project that detects road damage such as potholes and cracks using a YOLO object detection model.

The model is trained on a public road damage dataset and can detect road hazards in images and recorded driving videos by drawing bounding boxes around detected damage.

The project demonstrates transfer learning, object detection, computer vision, and deep learning techniques using YOLOv8.

---

## 🎯 Objectives

- Detect road damage using a YOLO object detection model.
- Classify multiple road damage categories.
- Evaluate model performance using standard object detection metrics.
- Demonstrate transfer learning for computer vision.
- Test the trained model on images and recorded road videos.
---

## 📂 Dataset
This project uses the public Road Damage Dataset (RDD2020).

The dataset contains over 26,000 road images collected from India, Japan, and the Czech Republic.

The model is trained to detect four categories of road damage:

- D00 – Longitudinal Crack
- D10 – Transverse Crack
- D20 – Alligator Crack
- D40 – Pothole

## 🖥️ Demo

(Add GIF or screenshots here)

Example:

Input:

📹 Road Video

↓

YOLO Detection

↓

🟩 Pothole Detected

↓

⚠️ Warning Displayed



## 🛠️ Technologies

- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- Google Colab
- VS Code
- Git
- GitHub
---

## 📂 Dataset

Dataset used:

- Road Damage Dataset (RDD2020)

The model was fine-tuned on publicly available road damage images.

---

## ⚙️ Project Pipeline

Road Video

↓

OpenCV

↓

YOLO Object Detection

↓

Hazard Detection

↓

Bounding Box

↓

Warning Display

---

## 📁 Project Structure

RoadGuardianAI/

├── app.py

├── detect.py

├── camera.py

├── warning.py

├── model/

│ └── best.pt

├── videos/

├── outputs/

├── utils/

├── requirements.txt

├── Dockerfile

└── README.md

---

## 📊 Model Evaluation

Metrics:

- Precision
- Recall
- mAP
- Inference Speed (FPS)

(Add your results after training.)

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/RoadGuardianAI.git

cd RoadGuardianAI

pip install -r requirements.txt
```

---

## ▶️ Running the Project

```bash
python app.py
```

---

## 📈 Future Improvements

- Voice warnings
- Distance estimation
- Detection of additional hazards:
  - road cracks
  - construction barriers
  - debris
  - puddles
- GPS logging
- Edge deployment (Jetson/Raspberry Pi)
- Night-time optimization

---

## 📚 What I Learned

- Object Detection with YOLO
- Computer Vision
- Transfer Learning
- OpenCV
- Real-time inference
- MLflow
- Docker
- End-to-end Machine Learning workflow

---

## 📄 License

This project is for educational purposes.
