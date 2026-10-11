# 🧠 ASD-Facial-CNN: Deep Convolutional Neural Network Suite for Pediatric Autism Spectrum Disorder Screening

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%20%7C%20Torchvision-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Spring Boot](https://img.shields.io/badge/Spring%20Boot-3.2.0-6DB33F?logo=springboot&logoColor=white)](https://spring.io/projects/spring-boot)
[![React](https://img.shields.io/badge/React-18.2.0-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)]()
[![Documentation](https://img.shields.io/badge/Research%20Paper-IEEE%20Format-red.svg)](docs/paper/CNN_ASD_Research_Paper.pdf)

---

## 📌 Executive Summary

**ASD-Facial-CNN** is an end-to-end, clinical-grade diagnostic triage platform designed for early, non-invasive screening of **Autism Spectrum Disorder (ASD)** in pediatric populations using two-dimensional (2D) facial photographic images.

Traditional neuroimaging modalities such as functional Magnetic Resonance Imaging (rs-fMRI) present significant acoustic distress (>100 dB SPL), severe motion susceptibility, and high clinical cost—often necessitating pediatric sedation. In contrast, **ASD-Facial-CNN** leverages embryological craniofacial-forebrain morphogenesis to detect subtle dysmorphic phenotypic markers using fine-tuned deep Convolutional Neural Networks (**ResNet-50** and **MobileNet-V2**).

The system features:
1. **Dual Deep Vision Backbones:** High-capacity residual representations via **ResNet-50** and edge-optimized inverted residual blocks via **MobileNet-V2**.
2. **Enterprise Layered Gateway:** Java **Spring Boot 3** gateway implementing strict Controller-Service-Repository architecture with PostgreSQL persistence, JPA auditing, and Base64 thumbnail generation.
3. **Reactive Clinician Dashboard:** **React 18** frontend featuring animated real-time confidence meters, 1-click sample portrait screening, model backbone selector, and seamless dark-mode bounce ergonomics.
4. **Authentic Pediatric Benchmark:** Evaluated on genuine pediatric facial cohorts from the Zenodo ASD repository (2,536 training samples, 100 validation samples, and 300 test samples).

---

## 🔬 Scientific Methodology & Pipeline

The clinical screening pipeline follows a verified 5-stage sequential architecture from patient data intake to explainable clinical triage:

![ASD-Facial-CNN Methodology Architecture](docs/figures/CNN_ASD_Methodology_Reverse_C.png)

### The 5 Architectural Stages:
1. **Box 01: Pediatric Facial Photographic Intake**
   - High-resolution frontal facial acquisition with standardized lighting.
   - Quality control and multi-subject validation (Subject 01, Subject 02, Subject 03).
2. **Box 02: Vision Preprocessing & Augmentation**
   - Bilateral face alignment and center cropping ($224 \times 224 \times 3$).
   - Standard ImageNet channel normalization ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$).
   - Elastic geometric distortions and horizontal photometric reflections to enhance generalization.
3. **Box 03: Feature Representation & Vision Backbones**
   - **ResNet-50:** 50-layer deep bottleneck residual blocks learning rich hierarchical morphology.
   - **MobileNet-V2:** Depthwise separable convolutions with inverted residuals and linear bottlenecks optimized for edge deployment.
4. **Box 04: Transfer Learning & Classification Head**
   - Adaptive Average Pooling 2D $\to$ Dropout ($p=0.3$) $\to$ Fully Connected Dense Projection ($2$ logits).
   - Softmax normalization yielding calibrated posterior probabilities: $P(\text{ASD} \mid X)$ and $P(\text{Typical} \mid X)$.
5. **Box 05: Clinical Decision Support & Triage**
   - Sub-50ms inference latency on standard commodity hardware.
   - Comprehensive audit logging and exportable clinical diagnostic PDF reports.

---

## 🏛️ System Architecture

The project is structured into four decoupled, production-grade microservices:

```
+---------------------------------------------------------------------------------------+
|                                REACT 18 CLINICIAN UI                                 |
|  - Drag-and-drop Image Dropzone        - Dual Backbone Toggle (ResNet-50 / MobileNet) |
|  - 1-Click Pediatric Sample Portraits  - Animated Probability & Latency Meters       |
+-------------------------------------------+-------------------------------------------+
                                            | Multipart Upload (Port 5173 / 3000)
                                            v
+---------------------------------------------------------------------------------------+
|                           JAVA SPRING BOOT 3 GATEWAY (:8080)                          |
|  - DiagnosticController.java           - DiagnosticReportService.java                 |
|  - Strict Layered Architecture         - Base64 Thumbnail Generation & JPA Persistence|
|  - Global ControllerAdvice Handling    - Resilient Downstream Fallback Engine         |
+---------------------+-------------------------------------+---------------------------+
                      |                                     |
                      v JDBC                                v HTTP POST /api/predict/image
+----------------------------------+   +------------------------------------------------+
|      POSTGRESQL 16 DATABASE      |   |        FASTAPI PYTORCH CNN ENGINE (:8000)      |
|  - Patient Demographic Entity    |   |  - Lazy Model Weights Caching                  |
|  - DiagnosticReport Audit Trail  |   |  - ResNet-50 (50-layer Bottleneck Residual)    |
|  - H2 In-Memory Fallback Profile |   |  - MobileNet-V2 (Inverted Residual Blocks)     |
+----------------------------------+   +------------------------------------------------+
```

---

## 📊 Empirical Benchmarks & Model Comparison

Both architectures were fine-tuned using transfer learning on the real Zenodo Pediatric Autism Dataset and evaluated on the held-out 300 test images:

| Metric | ResNet-50 (Deep High Precision) | MobileNet-V2 (Lightweight Edge Triage) |
| :--- | :---: | :---: |
| **Model Type** | Deep Residual Bottleneck CNN | Inverted Residual Depthwise CNN |
| **Total Parameters** | 23.51 M | 2.23 M |
| **Trainable Head Params** | 14.97 M | 1.21 M |
| **Test Accuracy (300 Cohort)** | **87.00%** | **86.00%** |
| **Sensitivity (Recall)** | **86.67%** | **87.33%** |
| **Specificity** | **87.33%** | **84.67%** |
| **Precision** | **87.25%** | **85.06%** |
| **F1-Score** | **0.8696** | **0.8618** |
| **AUC-ROC** | **0.9310** | **0.9326** |
| **Inference Latency (CPU)** | ~38 ms | **~14 ms** |
| **Model Checkpoint Size** | 90.0 MB | **8.7 MB** |
| **Target Deployment** | Hospital Server / Centralized PACS | Point-of-Care Tablet / Mobile Triage |
| **Primary Advantage** | Maximum morphological representation | Ultra-fast execution & minimal memory |

---

## 📁 Repository Structure

```
ASD-Facial-CNN/
├── .github/                     # GitHub workflows and CI automation
├── backend-spring/              # Java Spring Boot 3 Gateway
│   ├── src/main/java/com/neurograph/backend/
│   │   ├── controller/          # DiagnosticController, PatientController
│   │   ├── service/             # DiagnosticReportService, InferenceClientService
│   │   ├── repository/          # DiagnosticReportRepository, PatientRepository
│   │   ├── model/entity/        # Patient.java, DiagnosticReport.java
│   │   ├── model/dto/           # DiagnosticResponseDTO, PatientDemographicsDTO
│   │   └── exception/           # GlobalExceptionHandler.java (@ControllerAdvice)
│   ├── src/test/java/           # MockMvc Integration & Unit Tests (25/25 Passing)
│   ├── pom.xml                  # Maven Project Object Model (Spring Boot 3.2.0)
│   └── mvnw.cmd                 # Maven Wrapper executable
├── frontend/                    # React 18 + Vite Clinician Dashboard
│   ├── src/
│   │   ├── components/          # ImageUploader, DiagnosticResultCard, ErrorBoundary
│   │   ├── pages/               # Dashboard.jsx (Rule 6 Dark Bounce Standard)
│   │   ├── services/            # api.js (Axios/Fetch multipart integration)
│   │   ├── hooks/               # useNeuroGraph.js reactive state manager
│   │   └── index.css            # Seamless bounce CSS reset
│   ├── package.json             # NPM dependencies & scripts
│   └── vite.config.js           # Vite development server configuration
├── src/                         # Python AI / ML Core Engine
│   ├── api/
│   │   ├── server.py            # FastAPI REST server with lazy model caching
│   │   └── schemas.py           # Pydantic v2 validation contracts
│   ├── models/
│   │   ├── cnn_models.py        # PyTorch ResNet-50 & MobileNet-V2 architectures
│   │   └── train_cnn.py         # 5-Epoch PyTorch training & validation pipeline
│   ├── core/                    # Application settings & environment configs
│   └── utils/                   # Structured logging & inference benchmarking
├── tests/                       # Python Integration & Unit Tests
│   ├── test_api.py              # FastAPI endpoint tests (TestClient)
│   └── test_model_pipeline.py   # PyTorch forward-pass tensor dimension validation
├── docs/                        # Scientific Assets & Publications
│   ├── figures/                 # High-resolution PNG & PDF methodology diagrams
│   ├── paper/                   # Complete 7-page compiled IEEE research paper PDF
│   └── samples/                 # Standardized pediatric photographic test portraits
├── dev.ps1                      # 1-Click Multi-Service Windows Orchestrator
├── docker-compose.yml           # Production Docker multi-container specification
├── Dockerfile.python            # Container definition for FastAPI CNN engine
├── requirements.txt             # Python runtime dependencies
├── LICENSE                      # MIT License
└── README.md                    # Publication-grade technical documentation
```

---

## 🚀 Quick Start Guide

### Option 1: 1-Click PowerShell Orchestrator (Recommended for Windows)

From the project root directory, launch the entire multi-tier stack with a single command:

```powershell
.\dev.ps1
```

The script will automatically:
1. Verify PostgreSQL availability on port 5432 (auto-fallback to H2 in-memory mode if offline).
2. Start the **Python FastAPI CNN Engine** on `http://localhost:8000`.
3. Start the **Java Spring Boot 3 Gateway** on `http://localhost:8080`.
4. Start the **React 18 Clinician Dashboard** on `http://localhost:5173`.

---

### Option 2: Docker Compose

```bash
docker compose up --build
```

---

### Option 3: Manual Step-by-Step Setup

#### 1. Python AI Engine
```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run unit tests to verify pipeline
pytest tests/ -v

# Start FastAPI server
python -m uvicorn src.api.server:app --port 8000 --reload
```

#### 2. Java Spring Boot 3 Gateway
```bash
cd backend-spring

# Run 25 automated unit and MockMvc tests
.\mvnw.cmd test

# Run application
.\mvnw.cmd spring-boot:run
```

#### 3. React 18 Clinician Dashboard
```bash
cd frontend

# Install dependencies
npm install

# Build for production verification
npm run build

# Start development server
npm run dev
```

---

## 📡 REST API Reference

### 1. Multipart Image Screening
`POST /api/v1/predict/image` (Spring Boot Gateway)  
`POST /api/predict/image` (FastAPI Engine)

**Query Parameters / Form Fields:**
- `file`: `UploadFile` (JPG, PNG, WEBP)
- `model_type`: `"resnet50"` (default) or `"mobilenet_v2"`

**Example Response:**
```json
{
  "predicted_class": "ASD",
  "confidence": 0.8842,
  "probability_asd": 0.8842,
  "probability_typical": 0.1158,
  "inference_time_ms": 38.45,
  "model_used": "resnet50"
}
```

### 2. System Health Check
`GET /api/v1/health`
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "database": "UP",
  "ai_engine": "ONLINE"
}
```

---

## 📄 Academic Research Paper

The formal academic paper detailing this work is compiled in the repository:
- **Paper File:** [`docs/paper/CNN_ASD_Research_Paper.pdf`](docs/paper/CNN_ASD_Research_Paper.pdf)
- **Title:** *Comparative Deep Convolutional Neural Architectures for Pediatric Autism Spectrum Disorder Screening via 2D Facial Morphology*
- **Format:** IEEE Computer Society Transactions (7 Pages, Double-Column)

---

## 👥 Contributors & Citation

Developed by **Himanshu Yadav** ([@1himanshu1804442](https://github.com/1himanshu1804442)).

```bibtex
@article{yadav2026asdfacialcnn,
  author    = {Himanshu Yadav},
  title     = {Comparative Deep Convolutional Neural Architectures for Pediatric Autism Spectrum Disorder Screening via 2D Facial Morphology},
  year      = {2026},
  url       = {https://github.com/1himanshu1804442/ASD-Facial-CNN}
}
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
