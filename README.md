# 🧠 NeuroScan AI

<div align="center">

### AI-Powered Multi-Modal Medical Imaging Platform

**Brain MRI Tumor Detection • Chest X-Ray Pneumonia Detection • Explainable AI • Medical Reporting**

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-DeepLearning-D00000?style=for-the-badge&logo=keras)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-7952B3?style=for-the-badge&logo=bootstrap)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

---

*A modern AI-powered clinical decision support platform combining Deep Learning, Explainable AI, Medical Image Processing and Interactive Analytics for Brain MRI Tumor Detection and Chest X-Ray Pneumonia Detection.*

**GitHub Repository**

https://github.com/Kati1997/NeuroScan-AI

</div>

---

# 📑 Table of Contents

- Project Overview
- Features
- System Workflow
- Platform Architecture
- Technologies
- Deep Learning Models
- Explainable AI
- Performance
- Screenshots
- Installation
- Docker Deployment
- REST API
- Project Structure
- Security
- Academic Collaboration
- Contributors
- Future Improvements
- License

---

# 🏥 Project Overview

NeuroScan AI is an AI-powered medical imaging platform designed to assist healthcare professionals in the analysis of Brain MRI and Chest X-Ray images using state-of-the-art Deep Learning models.

The platform combines medical image classification, explainable artificial intelligence (Grad-CAM), automated PDF report generation, clinical analytics, secure authentication, and RESTful APIs into a single web application.

Unlike traditional image classification demos, NeuroScan AI provides a complete workflow that resembles a real clinical environment, including:

- Doctor authentication
- Secure patient image upload
- AI-assisted diagnosis
- Explainable AI visualization
- Prediction history
- Medical PDF reports
- Performance analytics
- Model comparison dashboard

The system was developed for academic research purposes while following software engineering practices used in real-world AI healthcare applications.

---

# ✨ Key Features

## 🧠 Brain MRI Analysis

- Brain tumor detection
- Multiple Deep Learning architectures
- Automatic best-model selection
- Confidence score calculation
- Grad-CAM visualization
- Prediction history
- PDF medical report generation

---

## 🫁 Chest X-Ray Analysis

- Pneumonia detection
- Fine-tuned MobileNetV2
- AI confidence estimation
- Explainable AI heatmaps
- Medical report generation
- Historical predictions

---

## 🔒 Secure Authentication

- Doctor registration
- Login system
- Password hashing
- Session management
- Role-based access control

---

## 📊 Analytics Dashboard

The platform includes an interactive dashboard showing:

- Total analyses
- MRI analyses
- Chest X-Ray analyses
- Tumor statistics
- Pneumonia statistics
- Healthy cases
- Confusion matrices
- Accuracy metrics
- Precision
- Recall
- F1 Score
- ROC AUC
- Daily analytics
- Monthly analytics

---

## 📄 Medical Reporting

Automatic PDF reports include:

- Diagnosis
- Confidence score
- Scan information
- Model used
- Date and time
- Doctor information
- Explainable AI visualization

---

## 🤖 Explainable AI

The platform integrates Grad-CAM visualizations allowing users to understand which image regions contributed most to the AI prediction.

This increases model transparency and improves trust in AI-assisted medical diagnosis.

---

## 🌐 REST API

RESTful JSON API supporting:

- Authentication
- Predictions
- Dashboard statistics
- Reports
- History
- Model information

---

## 🐳 Docker Support

The entire application can be deployed using Docker and Docker Compose with minimal configuration.

---

# ⚙️ System Workflow

```text
Doctor Login
      │
      ▼
Authentication
      │
      ▼
Select Scan Type
      │
      ├─────────────┐
      ▼             ▼
 Brain MRI      Chest X-Ray
      │             │
      ▼             ▼
 Upload Image
      │
      ▼
 Image Preprocessing
      │
      ▼
 Deep Learning Prediction
      │
      ▼
 Confidence Calculation
      │
      ▼
 Grad-CAM Generation
      │
      ▼
 Medical Diagnosis
      │
      ▼
 Save Prediction
      │
      ▼
 Generate PDF Report
      │
      ▼
 Analytics Dashboard
```

---

# 🏗️ Platform Architecture

```text
                     User
                      │
                      ▼
             Flask Web Application
                      │
      ┌───────────────┼────────────────┐
      ▼               ▼                ▼
 Authentication   Deep Learning     REST API
      │               │                │
      ▼               ▼                ▼
 SQLite DB      TensorFlow Models   JSON Services
      │               │
      ▼               ▼
 Prediction History
      │
      ▼
 PDF Reports
      │
      ▼
 Analytics Dashboard
```

---

# 💻 Technologies

| Category | Technologies |
|-----------|--------------|
| Backend | Flask, Python |
| Frontend | HTML5, CSS3, JavaScript, Bootstrap |
| AI Framework | TensorFlow, Keras |
| Database | SQLite |
| Computer Vision | OpenCV |
| Explainable AI | Grad-CAM |
| Charts | Chart.js |
| Reports | ReportLab |
| Deployment | Docker |
| API | REST JSON |

---

# 🧠 Deep Learning Models

The platform supports multiple Deep Learning architectures for medical image classification.

## Brain MRI

- Custom CNN
- ResNet50
- MobileNetV2

The application automatically compares the trained models and selects the best-performing model based on evaluation metrics.

---

## Chest X-Ray

- MobileNetV2 (Fine-Tuned)

Optimized for binary classification:

- Normal
- Pneumonia

---

The AI pipeline automatically performs:

- Image preprocessing
- Normalization
- Prediction
- Confidence estimation
- Grad-CAM generation
- Report generation

---
