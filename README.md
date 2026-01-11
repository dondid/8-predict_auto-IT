# 🚗 Advanced Auto Analytics Platform (2026 Edition)

Un sistem complet de inteligență artificială pentru piața auto, care combină **Machine Learning Predictiv**, **Analiză Financiară Live** și **Generative AI** pentru o perspectivă 360°.

![Project Status](https://img.shields.io/badge/Status-Complete-green)
![Python](https://img.shields.io/badge/Python-3.9-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

---

## 🌟 Ce face acest proiect?

Nu este doar un simplu script de predicție. Este o platformă "Enterprise-Grade" care răspunde la 3 întrebări critice:
1.  **Cât valorează?** (ML Prediction & Modern Comparison)
2.  **Cât de sigură/fiabilă este?** (Safety Score & NHTSA Data)
3.  **Este o investiție bună?** (Live Market Data & AI Expert Verdict)

---

## 🛠️ Arhitectura Tehnică

Proiectul este împărțit în module distincte, interconectate profesional:

### 1. 🧬 Core ML Engine (`src/models`)
- **Supervised**: Random Forest, XGBoost, SVR, Neural Networks.
- **Unsupervised**: K-Means Clustering (Segmentare Piață).
- **Training**: Antrenat pe dataset-ul UCI (1985) pentru precizie istorică.
- **Validare**: Cross-Validation (30 runs), Wilcoxon Test pentru comparație statistică.

### 2. 🧠 AI "Senior Analyst" (`src/ai`)
- **Tehnologie**: Google Gemini 1.5 Flash.
- **Rol**: Analist auto care primește contextul tehnic și financiar.
- **Capabilități**:
    - Generează rapoarte detaliate (istoric, probleme).
    - Estimează evoluția valorii (1985-2025).
    - Nu halucinează (are acces la date reale).

### 3. 🌐 Live & Modern Data Layer (`src/data`)
- **Yahoo Finance API**: Preia în timp real prețul acțiunilor (ex: BMW.DE) și știri financiare.
- **Modern Dataset (2024)**: Bază de date secundară cu mii de mașini moderne pentru comparație preț.
- **SQL Backend**: Stocarea datelor în SQLite (`automobile.db`) pentru persistență.

### 4. 📊 Dashboard Interactiv (`dashboard.py`)
- **Framework**: Streamlit.
- **Tab-uri**:
    - *Live Prediction*: Predicție preț + Safety Score.
    - *Brand Encyclopedia*: Rapoarte AI + Grafice Live Market.
    - *Data Explorer*: Vizualizări de distribuție + **Clustering Automat (K-Means)**.
    - *Export*: Generare automată Kit Prezentare (ZIP).

---

## 🎥 Galerie & Demo
 
### Video Demonstrativ
Prezentare completă a funcționalităților

https://github.com/user-attachments/assets/fc2d0474-9189-49b0-a0e4-da80e42bf5cf

## 🚀 Cum rulezi proiectul?

### Varianta A: Docker (Recomandat)
Scapi de configurări manuale. Totul e izolat.
1. Configurează `.env` cu cheia ta Gemini.
2. Rulează:
   ```bash
   docker-compose up --build
   ```
3. Deschide `http://localhost:8501`.

### Varianta B: Local (PowerShell/Terminal)
1. **Instalare**:
   ```bash
   pip install -r environment.yml
   ```
2. **Download Date Noi** (dacă nu există):
   ```bash
   python -m scripts.download_modern_data
   ```
3. **Pornire**:
   ```bash
   streamlit run dashboard.py
   ```

---

## 📈 Cum generezi materialele pentru prezentare?

Dacă vrei graficele pentru PowerPoint sau Licență:

1. **Din Interfață**:
   - Deschide Dashboard-ul -> Sidebar Stânga jos.
   - Apasă **"📦 Descarcă Resurse"**.
   - Primești un ZIP cu: Ghidul de prezentare, Tabele CSV, Grafice.

2. **Din Linie de Comandă** (pentru grafice tehnice):
   - Rulează scriptul de generare plot-uri (Feature Importance, Radar Chart):
   ```bash
   python -m scripts.generate_presentation_plots
   ```
   - Găsești fișierele PNG în `outputs/figures/`.

---

## 📁 Structura Fișierelor

```
📂 predict_auto/
├── 📄 dashboard.py            # Punctul central (Interfața)
├── 📄 .env                    # Chei API (Secret!)
├── 📂 src/
│   ├── 🧠 ai/                 # gemini_service.py (Creierul AI)
│   ├── 🌐 data/               # live_api.py (Yahoo), loader.py (SQL/CSV)
│   ├── 🤖 models/             # modelel ML salvate
│   └── ⚙️ evaluation/         # statistical_tests.py
├── 📂 outputs/
│   ├── 📉 figures/            # Graficele salvate (PNG)
│   └── 📑 reports/            # Rapoarte CSV
└── 📂 scripts/                # Utilitare (download, plot generator)
```

---

## 📸 Galerie Rezultate

### 1. Factori Determinanți (Feature Importance)
Ce contează cel mai mult în stabilirea prețului? (Analiză XGBoost/Random Forest)

<img width="1000" height="600" alt="presentation_feature_importance" src="https://github.com/user-attachments/assets/2a6d21a7-da7f-4d2c-8cd3-56b882d0ecc3" />


### 2. Performanța Modelelor (R²)
Comparație directă între algoritmii testați.

<img width="1000" height="600" alt="presentation_model_comparison" src="https://github.com/user-attachments/assets/93ad6ac0-7c67-4e54-a9b7-8fb12eed9ba7" />


### 3. Validare Statistică (Wilcoxon Heatmap)
Dovada științifică că diferențele dintre modele sunt semnificative (p < 0.05).

<img width="3000" height="2400" alt="wilcoxon_heatmap" src="https://github.com/user-attachments/assets/613788f8-38a2-44e4-a4e7-f9f880e34a01" />


---
*Acest proiect demonstrează competențe Full-Stack Data Science: de la ETL și SQL, la ML avansat și integrare LLM în producție.*