# AI-Powered Brain Tumor Detection and Medical Decision Support System

A full-stack web application for medical image processing and brain tumor detection using Deep Learning, Computer Vision, and Explainable AI (Grad-CAM).

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple)

## Features

- **Secure Authentication** — Doctor/Admin registration, login, password hashing, session management
- **Deep Learning Models** — Custom CNN, ResNet50, MobileNetV2 with automatic best-model selection
- **MRI Analysis** — Upload, preview, and analyze brain MRI scans
- **Explainable AI** — Grad-CAM heatmap visualizations
- **PDF Reports** — Automatic hospital-style medical report generation
- **Analytics Dashboard** — Chart.js powered real-time statistics
- **REST API** — Full JSON API with validation and documentation
- **Docker Deployment** — One-command deployment with docker-compose

## Project Structure

```
BrainTumorDetectionAI/
├── app.py                  # Main Flask application
├── api.py                  # REST API endpoints
├── database.py             # SQLite database operations
├── train_model.py          # Train all 3 DL models
├── compare_models.py       # Model comparison utilities
├── predict.py              # Prediction + Grad-CAM + PDF reports
├── create_demo_model.py    # Quick demo model for testing
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── models/                 # Trained model files
├── uploads/                # Uploaded MRI images
├── reports/                # Generated PDF reports
├── static/                 # CSS, JS, Grad-CAM images
├── templates/              # HTML templates
└── database/               # SQLite DB + schema
```

## Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd BrainTumorDetectionAI

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
cp .env.example .env
# Edit .env and set FLASK_SECRET_KEY to a random string
```

### 3. Create Demo Model (Quick Test)

For immediate testing without the Kaggle dataset:

```bash
python create_demo_model.py
```

### 4. Train with Real Dataset (Recommended)

Download the [Brain MRI dataset from Kaggle](https://www.kaggle.com/code/mahmoudmagdyelnahal/brain-mri-images-for-brain-tumor-detection) and organize as:

```
dataset/
├── no_tumor/
│   └── *.jpg
└── tumor/
    └── *.jpg
```

Then train all models:

```bash
python train_model.py --data-dir dataset
python compare_models.py
```

### 5. Run the Application

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

**Default Admin Account:**
- Email: `admin@hospital.com`
- Password: `Admin@123`

## Docker Deployment

### One-Command Deployment

```bash
docker-compose up --build
```

Access at **http://localhost:5000**

### Build Manually

```bash
docker build -t brain-tumor-detection .
docker run -p 5000:5000 -e FLASK_SECRET_KEY=your-secret brain-tumor-detection
```

## Deploy to Render

1. Push your code to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your repository
4. Configure:
   - **Build Command:** `pip install -r requirements.txt && python create_demo_model.py`
   - **Start Command:** `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 app:app`
   - **Environment Variables:**
     - `FLASK_SECRET_KEY` = (generate a random string)
     - `FLASK_ENV` = `production`
5. Add a **Persistent Disk** mounted at `/app/database`, `/app/uploads`, `/app/reports`, `/app/models`
6. Deploy

## Deploy to Railway

1. Push your code to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Railway auto-detects the Dockerfile
4. Set environment variables:
   - `FLASK_SECRET_KEY`
   - `FLASK_ENV=production`
   - `PORT` (Railway sets this automatically)
5. Add volumes for persistent storage (database, uploads, reports, models)
6. Deploy

## REST API Documentation

Base URL: `/api`

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/register` | No | Register new doctor |
| POST | `/api/login` | No | Authenticate user |
| POST | `/api/logout` | Yes | End session |
| POST | `/api/upload` | Yes | Upload MRI image |
| POST | `/api/predict` | Yes | Analyze MRI + generate report |
| GET | `/api/predictions` | Yes | List prediction history |
| GET | `/api/reports` | Yes | List PDF reports |
| GET | `/api/dashboard` | Yes | Analytics data |
| GET | `/api/docs` | No | API documentation |

### Example: Login

```bash
curl -X POST http://localhost:5000/api/login \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@hospital.com", "password": "Admin@123"}'
```

### Example: Predict

```bash
curl -X POST http://localhost:5000/api/predict \
  -b cookies.txt \
  -F "file=@mri_scan.jpg"
```

## Database Schema

See `database/schema.sql` for the full SQL schema.

**Tables:**
- `users` — Doctor/Admin accounts
- `predictions` — MRI analysis results
- `reports` — Generated PDF report references

## Model Training Details

Three models are trained and compared:

| Model | Type | Key Features |
|-------|------|-------------|
| Custom CNN | From Scratch | 4 conv layers, batch norm, dropout |
| ResNet50 | Transfer Learning | ImageNet pre-trained, fine-tuned head |
| MobileNetV2 | Transfer Learning | Lightweight, fast inference |

**Evaluation Metrics:** Accuracy, Precision, Recall, F1 Score, ROC/AUC, Confusion Matrix

The best model (by F1 Score) is automatically saved as `models/best_model.keras`.

## Security Features

- Password hashing (Werkzeug)
- CSRF protection (Flask-WTF)
- SQL injection protection (parameterized queries)
- Secure file upload validation
- File size limits (10MB default)
- HTTP-only session cookies
- Role-based access control

## Pages

| Page | URL | Access |
|------|-----|--------|
| Home | `/` | Public |
| About | `/about` | Public |
| Login | `/login` | Public |
| Register | `/register` | Public |
| Dashboard | `/dashboard` | Authenticated |
| Upload MRI | `/upload` | Authenticated |
| Predictions | `/predictions` | Authenticated |
| Reports | `/reports` | Authenticated |
| Model Comparison | `/model-comparison` | Authenticated |

## Medical Disclaimer

This system is designed for **educational and research purposes** as a university final project and portfolio piece. It should **not** be used as the sole basis for clinical diagnosis. Always consult qualified medical professionals for medical decisions.

## License

MIT License — Free for educational and research use.
