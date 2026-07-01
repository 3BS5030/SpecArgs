<p align="center">
  <img src="https://img.shields.io/badge/Laravel-12-FF2D20?style=for-the-badge&logo=laravel&logoColor=white" alt="Laravel 12">
  <img src="https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch 2.6">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/Flask-3.1-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask 3.1">
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/Bootstrap_5-7952B3?style=for-the-badge&logo=bootstrap&logoColor=white" alt="Bootstrap 5">
  <img src="https://img.shields.io/badge/Tailwind_CSS-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white" alt="Tailwind CSS">
</p>

<p align="center">
  <img src="https://img.shields.io/github/license/your-username/brain-tumor" alt="License">
  <img src="https://img.shields.io/github/stars/your-username/brain-tumor?style=social" alt="Stars">
  <img src="https://img.shields.io/github/forks/your-username/brain-tumor?style=social" alt="Forks">
</p>

---

<h1 align="center">Brain Tumor MRI Classification</h1>
<p align="center">
  <strong>AI-Powered Brain Tumor Detection from MRI Scans</strong><br>
  Upload MRI images and receive instant classification results with confidence scores.<br>
  Built with Laravel + PyTorch, featuring bilingual support (English/Arabic) and a modern clinical-grade UI.
</p>

---

## Features

| Feature | Description |
|---------|-------------|
| **MRI Scan Upload** | Upload JPG/PNG images securely via the web interface |
| **AI Classification** | PyTorch CNN model classifies tumors into 4 categories |
| **Confidence Scores** | Per-class probability breakdown with visual progress bars |
| **Prediction History** | Full history dashboard tied to authenticated users |
| **User Authentication** | Secure registration/login with session-based auth |
| **Bilingual UI** | Full English and Arabic support with RTL layout |
| **Dark / Light Theme** | Toggle between themes, persisted in localStorage |
| **Docker Ready** | One-command deploy with Docker + Railway support |

### Classifiable Tumor Types

| Label | Description |
|-------|-------------|
| Glioma | Tumor originating from glial cells |
| Meningioma | Tumor arising from the meninges |
| Pituitary | Pituitary gland tumor |
| No Tumor | Healthy brain scan |

---

## Architecture

```
+------------------+        +--------------------------------+
|                  |  HTTP  |                                |
|   Laravel 12     |<------>|   Flask Inference Service      |
|   (Web + Auth)   |        |   (PyTorch CNN Model)          |
|                  |        |                                |
+------------------+        +--------------------------------+
         |                            |
         v                            v
   +----------+               +------------------+
   | SQLite / |               | best_model       |
   | Postgres |               | .pth weights     |
   +----------+               +------------------+
```

The system uses a **two-service architecture**:
- **Laravel** handles user authentication, file uploads, prediction history, and the web UI
- **Flask + PyTorch** runs the deep learning inference as a dedicated microservice
- Communication occurs over HTTP -- the Laravel app sends the image to Flask and receives predictions

### Neural Network

The classifier is a **Convolutional Neural Network (CNN)** built with PyTorch:
- 2 convolutional layers (ReLU + MaxPool)
- 2 fully connected layers
- Input: 224x224 RGB images
- Normalization: ImageNet mean/std

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Laravel 12 (PHP 8.2) |
| **Frontend** | Blade + Bootstrap 5 + Tailwind CSS 4 |
| **ML Engine** | PyTorch 2.6 (CPU) |
| **ML API** | Flask 3.1 |
| **Database** | SQLite (dev) / PostgreSQL (prod) |
| **Build Tool** | Vite 7 + Laravel Vite Plugin |
| **Container** | Docker + Supervisor |
| **Cloud** | Railway ready |

---

## Getting Started

### Prerequisites

- PHP ^8.2
- Composer 2
- Node.js ^20
- Python 3.8+

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/brain-tumor.git
cd brain-tumor

# 2. Install PHP dependencies
composer install

# 3. Install Node dependencies
npm install

# 4. Environment setup
cp .env.example .env
php artisan key:generate

# 5. Run database migrations
php artisan migrate

# 6. Install Python dependencies
cd app/Infrastructure/Prediction/Python
pip install -r requirements.txt
cd ../../../../

# 7. Start the Flask inference service
python app/Infrastructure/Prediction/Python/prediction_server.py &

# 8. Build frontend assets
npm run build

# 9. Start Laravel dev server
php artisan serve
```

### Docker Deploy

```bash
docker build -t brain-tumor .
docker run -p 8080:80 brain-tumor
```

---

## Usage

1. **Register** an account at `/register`
2. **Log in** at `/login`
3. Navigate to **Scan** and upload an MRI image
4. View the **prediction result** with per-class confidence scores
5. Access your full **prediction history** from the dashboard

---

## Project Structure

```
app/
+-- Application/Prediction/UseCases/     # Application business logic
+-- Domain/Prediction/Contracts/          # Domain interfaces
+-- Http/
|   +-- Controllers/                      # Auth, Dashboard, Locale
|   +-- Middleware/                       # Locale middleware
+-- Infrastructure/
|   +-- Persistence/Prediction/           # Eloquent repositories
|   +-- Prediction/Python/               # Flask server + PyTorch model
+-- Models/                               # User, PredictionHistory
+-- Providers/                            # AppServiceProvider
resources/
+-- lang/{en,ar}/                        # Bilingual translations
+-- views/                               # Blade templates
routes/web.php                            # All web routes
config/brain_tumor.php                    # Prediction service config
```

---

## Configuration

Key environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `BRAIN_TUMOR_SERVICE_URL` | `http://127.0.0.1:5001` | Flask service URL |
| `BRAIN_TUMOR_MODEL_PATH` | `app/.../best_model.pth` | Path to model weights |
| `BRAIN_TUMOR_IMAGE_SIZE` | `224` | Input image size (px) |
| `BRAIN_TUMOR_CLASS_LABELS` | `glioma,meningioma,notumor,pituitary` | Classification labels |

---

## Roadmap

- [x] User authentication
- [x] MRI upload and classification
- [x] Prediction history dashboard
- [x] Bilingual (EN/AR) support
- [x] Dark/light theme
- [x] Docker deployment
- [ ] Additional model architectures (ResNet, EfficientNet)
- [ ] Batch upload support
- [ ] API rate limiting
- [ ] Patient report PDF export
- [ ] Integration with DICOM viewers

---

## License

This project is open-sourced software licensed under the [MIT license](LICENSE).

---

<p align="center">
  <sub>Built with dedication to advancing medical AI diagnostics.</sub>
</p>
