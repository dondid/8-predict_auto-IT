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

## 🛠️ Arhitectura Tehnică & Module Noi

Proiectul este împărțit în module distincte, interconectate profesional:

### 1. 📈 Financial Analysis & Signals (`src/financial`) [NOU]
- **Market Data**: Preia date live de pe bursă (Yahoo Finance) pentru 45+ companii auto.
- **Advanced Models**: Include modele Deep Learning pentru predicția trendului bursier:
  - **LSTM, GRU, RNN**: Rețele recurente pentru serii de timp.
  - **TCN & Transformer**: Arhitecturi state-of-the-art pentru secvențe.
- **PDF Reporting**: Generare automată de rapoarte PDF cu grafice și analize.

### 2. 🧠 Multi-Model AI Assistant (`src/ai`)
- **Chat Avansat**: Asistent virtual cu personalități multiple:
  - **Gemini 1.5** (Google) - Online, rapid.
  - **Grok** (xAI) - Online, expert tehnic și creativ.
  - **GPT-2** (Local) - Offline, rulează pe CPU.
- **Analiză Semantică**: Interpretează datele financiare și oferă sfaturi de investiții.

### 3. 🧬 Core ML Engine (`src/models`)
- **Supervised**: Random Forest, XGBoost, SVR, Neural Networks pentru prețul mașinilor.
- **Unsupervised**: K-Means Clustering pentru segmentarea pieței.

### 4. 📊 Premium Dashboard (`dashboard.py`)
- **Framework**: Streamlit cu temă **Dark Corporate** personalizată.
- **Pagini Cheie**:
    - *Financial Analysis*: Grafice interactive, Dropdown selecție companii, Semnale BUY/SELL.
    - *AI Assistant*: Chat liber cu alegerea modelului de inteligență.
    - *Live Prediction*: Estimare preț mașini SH.

---

## 🚀 Cum rulezi proiectul?

### Varianta A: Docker (Recomandat)
Scapi de configurări manuale. Totul e izolat.
1. Configurează `.env` cu cheile tale (GEMINI_API_KEY, GROK_API_KEY).
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
2. **Pornire**:
   ```bash
   streamlit run dashboard.py
   ```

---

## 📸 Galerie Rezultate

### 1. Financial Analysis Page
Interfață profesională cu grafice de acțiuni, indicatori de volatilitate și tabel clar de predicții ML. Include generare raport PDF.

### 2. AI Assistant (Multi-Model)
Posibilitatea de a discuta cu Grok, Gemini sau GPT-2 direct din interfață.

### 3. Factori Determinanți (Feature Importance)
Analiză XGBoost/Random Forest asupra prețului.

<img width="1000" height="600" alt="presentation_feature_importance" src="https://github.com/user-attachments/assets/2a6d21a7-da7f-4d2c-8cd3-56b882d0ecc3" />


### 2. Performanța Modelelor (R²)
Comparație directă între algoritmii testați.

<img width="1000" height="600" alt="presentation_model_comparison" src="https://github.com/user-attachments/assets/93ad6ac0-7c67-4e54-a9b7-8fb12eed9ba7" />


### 3. Validare Statistică (Wilcoxon Heatmap)
Dovada științifică că diferențele dintre modele sunt semnificative (p < 0.05).

<img width="3000" height="2400" alt="wilcoxon_heatmap" src="https://github.com/user-attachments/assets/613788f8-38a2-44e4-a4e7-f9f880e34a01" />


---
*Acest proiect demonstrează competențe Full-Stack Data Science: de la ETL și SQL, la ML avansat și integrare LLM în producție.*