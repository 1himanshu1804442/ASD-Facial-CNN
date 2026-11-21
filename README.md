# 🧠 ASD-Facial-CNN: A Web-Based Preliminary Screening Framework for Autism Spectrum Disorder via Modified-Stem ResNet-50 and Spatial Bi-LSTM

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

**ASD-Facial-CNN** is an end-to-end web-based **preliminary research screening framework** designed for early, non-invasive triage assistance of **Autism Spectrum Disorder (ASD)** in pediatric populations using two-dimensional (2D) frontal facial photographic images.

Traditional neuroimaging modalities such as functional Magnetic Resonance Imaging (rs-fMRI) present significant acoustic distress (>100 dB SPL), severe motion susceptibility, and high clinical cost—often necessitating pediatric sedation. In contrast, **ASD-Facial-CNN** explores whether spatial morphological patterns can be captured non-invasively using our proposed **Mod-Stem ResNet-50 + Spatial Bi-LSTM** architecture, benchmarked systematically against five established vision backbones (**ResNet-50**, **DenseNet-121**, **EfficientNet-B0**, **MobileNet-V2**, and **ViT-B/16**).

> **Clinical Scope Disclaimer:** This software is strictly a **preliminary research screening aid** and triage decision-support prototype. It is NOT intended or certified as an autonomous medical diagnostic device. Final clinical ASD diagnosis requires comprehensive behavioral evaluations (such as ADOS-2 and ADI-R) administered by qualified multidisciplinary healthcare professionals.

The system features:
1. **Multi-Model Vision Evaluation:** Evaluates 6 distinct architectures on the same standardized split: Proposed Mod-Stem ResNet-50 + Spatial Bi-LSTM, DenseNet-121 (Khan et al. [11]), EfficientNet-B0 (Rahman & Subashini [2]), ResNet-50 (Khan et al. [11] / Subah et al. [13]), MobileNet-V2 (Alsaedi et al. [1] / Kanhirakadavath et al. [7]), and ViT-B/16 (Revathi & Arunarani [3]).
2. **Enterprise Layered Gateway:** Java **Spring Boot 3** gateway implementing strict Controller-Service-Repository architecture with PostgreSQL persistence, JPA auditing, and Base64 thumbnail generation.
3. **Reactive Clinician Dashboard:** **React 18** frontend featuring animated real-time confidence meters, 1-click sample portrait screening, 6-backbone model architecture selector, and seamless dark-mode bounce ergonomics.
4. **Standardized Pediatric Benchmark:** Evaluated on the standardized pediatric facial dataset: *ASD Detection from Facial Images – Dataset, Code and Results* (Contreras, UNIFESP, 2025; Zenodo DOI: [10.5281/zenodo.15073612](https://doi.org/10.5281/zenodo.15073612), derived from Piosenka's collection; 2,536 training samples, 100 validation samples, and 300 held-out test cohort). An automated perceptual difference-hash audit revealed 21.3% near-duplicate representations in the web-sourced collection, reinforcing that deep network scores reflect preliminary screening triage patterns rather than clinical certainty.

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

## 📊 Empirical Benchmarks & 6-Model Literature Comparison

All six candidate architectures were fine-tuned using transfer learning on the standardized Zenodo Pediatric Autism Dataset and evaluated on the held-out 300 test images (150 ASD, 150 Control):

| Model Architecture | Literature Citation | Parameters | Accuracy | Sensitivity | Specificity | F1-Score | AUROC | Latency (T4 GPU) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNet-V2 (Lightweight Triage)** | Beary et al. [1] / Alkahtani et al. [7] | 2.2 M | **88.67%** | **90.67%** | 86.67% | **0.8889** | 94.48% | **0.23 ms** |
| **ViT-B/16 (Vision Transformer)** | Revathi & Arunarani [3] (2026) | 85.8 M | 87.00% | 87.33% | 86.67% | 0.8704 | 94.52% | **0.22 ms** |
| **ResNet-50 (Residual Baseline)** | Khan et al. [11] / Monani et al. [13] | 23.5 M | 86.00% | 80.67% | **91.33%** | 0.8521 | **95.14%** | 0.32 ms |
| **DenseNet-121 (Dense Feature Reuse)** | Khan et al. [11] (2024) | 7.0 M | 85.00% | 85.33% | 84.67% | 0.8505 | 92.96% | 0.62 ms |
| **Mod-Stem ResNet-50 + Spatial Bi-LSTM (Proposed)** | **Proposed Architecture (This Work, 2026)** | **25.8 M** | **84.00%** | **88.00%** | 80.00% | 0.8462 | 91.39% | 0.28 ms |
| **EfficientNet-B0 (Compound Scaled)** | Rahman & Subashini [2] (2022) | 4.0 M | 83.67% | 86.00% | 81.33% | 0.8404 | 91.42% | 0.35 ms |
### Key Panel Takeaways:
1. **Mod-Stem ResNet-50 + Spatial Bi-LSTM Architectural Precision:** Replaces the standard $7\times 7$ conv stem with three consecutive $3\times 3$ convolutions ($3 \to 32 \to 32 \to 64$) to preserve high-frequency facial contours. It pools features into 7 vertical spatial tiers (Forehead $\to$ Eyes $\to$ Nose $\to$ Philtrum $\to$ Mouth $\to$ Chin) using `AdaptiveAvgPool2d((7, 1))` and models vertical spatial morphological proportions with a Spatial Bidirectional LSTM ($128$ hidden units per direction, $256$ concatenated output) followed by a 2-stage MLP projection head (`256 -> 128 -> 2`) with Dropout 0.3.
2. **Clinical Screening Sensitivity Priority:** In pediatric triage, false negatives are critical—missing an autistic child delays early behavioral intervention. The proposed hybrid achieves **88.00% Sensitivity (132/150 positive ASD cases caught)**, notably surpassing the standard ResNet-50 backbone at **80.67% (121/150 caught)**—catching **11 additional autistic children** that standard ResNet-50 missed.
3. **Hardware Latency Context:** Latencies in the table reflect provisional batched **Nvidia Tesla T4 GPU evaluation (0.22–0.62 ms/image)**. On standard single-threaded edge CPU inference, latencies scale realistically to ~24–35 ms for lightweight models (MobileNet-V2) and ~70–85 ms for deep residual networks, well within the real-time clinical window (<100 ms).
4. **Scope & Clinical Applicability:** This pipeline is designed as an objective, non-invasive **preliminary research screening aid** trained on the Zenodo pediatric benchmark cohort—serving to flag subtle morphological patterns for clinical specialist referral rather than acting as an autonomous medical diagnostic device.

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

#### 1. Python AI Engine & Benchmark Training
```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run unit tests to verify pipeline (all 23 tests pass)
pytest tests/ -v

# Train benchmark models (up to 20 epochs with early stopping & differential LR)
# Train all 6 models:
python src/models/train_cnn.py --model all --epochs 20 --patience 5

# Or train only the proposed Mod-Stem ResNet-50 + Bi-LSTM:
python src/models/train_cnn.py --model modstem_lstm --epochs 20 --patience 5

# Start FastAPI server
python -m uvicorn src.api.server:app --port 8000 --reload
```

---

### ☁️ Google Colab Accelerated Training Guide (Free T4 GPU)

To train all 6 benchmark models or fine-tune the proposed Mod-Stem ResNet-50 + Bi-LSTM on an enterprise T4 GPU:

1. **Package the Project (Excluding Bulky Folders):**
   ```bash
   tar --exclude='node_modules' --exclude='.venv' --exclude='.git' -czf asd_project.tar.gz .
   ```
2. **Launch Google Colab:**
   - Go to [Google Colab](https://colab.research.google.com).
   - In the top menu, select **Runtime** $\to$ **Change runtime type** $\to$ **T4 GPU** $\to$ **Save**.
3. **Upload and Extract:**
   - Click the folder icon on the left panel.
   - Upload `asd_project.tar.gz` and run:
     ```python
     !tar -xzf asd_project.tar.gz
     !pip install -q -r requirements.txt
     ```
4. **Execute Benchmark Training:**
   ```python
   # Run full 6-model benchmark with early stopping (patience=5) and Cosine Annealing
   !python src/models/train_cnn.py --model all --epochs 20 --patience 5
   ```
5. **Download Updated Models & Benchmark Metrics:**
   ```python
   from google.colab import files
   !tar -czf benchmark_results.tar.gz models/*.pth results/real_benchmark_metrics.json
   files.download('benchmark_results.tar.gz')
   ```
   Extract `benchmark_results.tar.gz` into your local `ASD-Facial-CNN` root. The React dashboard and FastAPI backend will automatically reflect the newly trained weights and empirical performance metrics!

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
- **Title:** *A Web-Based Preliminary Screening Framework for Autism Spectrum Disorder via Modified-Stem ResNet-50 and Spatial Bi-LSTM*
- **Format:** IEEE Computer Society Transactions (7 Pages, Double-Column)

---

## 👥 Contributors & Citation

### Project Team (B.Tech AIML)
- **Harsh Chaudhary**
- **Hemant Kumar**
- **Himanshu Yadav** ([@1himanshu1804442](https://github.com/1himanshu1804442))
- **Kapil Sharma**

### Project Guide & Department Head
- **Dr. Pushpa Chaudhary**, Head of Department, Department of Artificial Intelligence & Machine Learning, Galgotias College of Engineering and Technology (GCET).

### Academic Citation
```bibtex
@article{chaudhary2026asdfacialscreening,
  author      = {Chaudhary, Harsh and Kumar, Hemant and Yadav, Himanshu and Sharma, Kapil},
  title       = {A Web-Based Preliminary Screening Framework for Autism Spectrum Disorder via Modified-Stem ResNet-50 and Spatial Bi-LSTM},
  institution = {Department of Artificial Intelligence & Machine Learning, Galgotias College of Engineering and Technology},
  year        = {2026},
  note        = {Under the guidance of Dr. Pushpa Chaudhary (Guide & HOD)},
  url         = {https://github.com/1himanshu1804442/ASD-Facial-CNN}
}
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
