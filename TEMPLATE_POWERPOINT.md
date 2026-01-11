# 📊 Template PowerPoint - Structură Detaliată

## 🎨 Design General
- **Font**: Arial sau Calibri, 24-28pt pentru titluri, 18-20pt pentru text
- **Culori**: 
  - Fundal: Alb sau albastru închis profesional
  - Text: Negru pe alb / Alb pe albastru
  - Accent: Albastru, portocaliu, verde (pentru grafice)
- **Layout**: Curat, spațios, fără aglomerare

---

## Slide 1: Title Slide 🎯

### Layout:
```
┌────────────────────────────────────────────┐
│                                            │
│     PREDICȚIA PREȚULUI AUTOMOBILELOR      │
│        Analiză Comparativă Multi-Model    │
│                                            │
│                 [Logo UCT]                 │
│                                            │
│              [Numele Tău]                  │
│           Machine Learning 2025            │
│                                            │
│           14 Ianuarie 2025, 14:00          │
│                                            │
└────────────────────────────────────────────┘
```

### Conținut Exact:
```
Titlu Principal:
PREDICȚIA PREȚULUI AUTOMOBILELOR

Subtitlu:
Analiză Comparativă Multi-Model

Autor:
[Numele Tău Complet]
Grupa [X]

Curs:
Machine Learning
Prof. Dr. Ruxandra Stoean

Data:
14 Ianuarie 2025
```

---

## Slide 2: Introducere 📋

### Layout:
```
┌────────────────────────────────────────────┐
│  Introducere                               │
│                                            │
│  PROBLEMA                                  │
│  • Predicția prețului automobile          │
│  • Pe baza caracteristicilor tehnice      │
│                                            │
│  DATASET                                   │
│  • UCI Automobile Data Set                │
│  • 205 instanțe, 26 atribute              │
│  • Features: engine-size, horsepower...   │
│                                            │
│  MOTIVAȚIE                                 │
│  • Platforme vânzare auto                 │
│  • Evaluare vehicule second-hand          │
│  • Decizie automată credit auto           │
│                                            │
│  [Mini plot: price distribution]           │
└────────────────────────────────────────────┘
```

### Imagini de inclus:
- `price_distribution.png` (partea din stânga - histograma)

---

## Slide 3: Metodologie 🔄

### Layout:
```
┌────────────────────────────────────────────┐
│  Pipeline Machine Learning                 │
│                                            │
│         ┌─────────────┐                   │
│         │    Date     │                   │
│         └──────┬──────┘                   │
│                ↓                           │
│         ┌─────────────┐                   │
│         │Preprocessing│                   │
│         └──────┬──────┘                   │
│                ↓                           │
│      ┌─────────────────┐                  │
│      │     Modele      │                  │
│      │ RF│XGB│SVR│NN  │                  │
│      └────────┬────────┘                  │
│               ↓                            │
│         ┌─────────────┐                   │
│         │  Evaluare   │                   │
│         └──────┬──────┘                   │
│                ↓                           │
│         ┌─────────────┐                   │
│         │ Comparație  │                   │
│         └─────────────┘                   │
│                                            │
│  Preprocessing:                            │
│  • Cleaning, Encoding, Scaling             │
│  • Feature Engineering                     │
│                                            │
│  Evaluare:                                 │
│  • MSE, RMSE, MAE, R²                     │
│  • Cross-Validation (30 runs)             │
│  • Wilcoxon Test                           │
└────────────────────────────────────────────┘
```

### Elemente Cheie:
- Diagramă flow vizuală
- Bullets scurți pentru fiecare etapă
- Folosește iconițe sau forme colorate

---

## Slide 4: Random Forest 🌲

### Layout:
```
┌────────────────────────────────────────────┐
│  Model 1: Random Forest Regressor         │
│                                            │
│  CONFIGURAȚIE                              │
│  • Ensemble: 200 arbori                   │
│  • Max depth: 20                           │
│  • Bagging cu replacement                 │
│                                            │
│  ┌──────────────────┬──────────────────┐  │
│  │ Feature Import.  │  Predictions     │  │
│  │                  │                  │  │
│  │  [PLOT 1]        │   [PLOT 2]       │  │
│  │                  │                  │  │
│  │                  │                  │  │
│  └──────────────────┴──────────────────┘  │
│                                            │
│  REZULTATE TEST SET                        │
│  • R² = [X.XXXX]                          │
│  • RMSE = [X,XXX.XX]                      │
│  • MAE = [X,XXX.XX]                       │
│                                            │
│  Top Features: engine-size, curb-weight   │
└────────────────────────────────────────────┘
```

### Imagini de inclus:
- **Stânga**: `rf_feature_importance.png`
- **Dreapta**: `rf_predictions.png` (partea pentru Test Set)

### Textbox pentru metrici:
```
┌─────────────────────────┐
│ REZULTATE TEST SET      │
│ ─────────────────────── │
│ R² Score:  0.XXXX       │
│ RMSE:      X,XXX.XX $   │
│ MAE:       X,XXX.XX $   │
│ MAPE:      XX.XX %      │
└─────────────────────────┘
```

---

## Slide 5: XGBoost 🚀

### Layout:
```
┌────────────────────────────────────────────┐
│  Model 2: XGBoost (Gradient Boosting)     │
│                                            │
│  CONFIGURAȚIE                              │
│  • Boosting: 200 estimatori               │
│  • Learning rate: 0.1                      │
│  • Early stopping activat                 │
│                                            │
│  ┌──────────────────┬──────────────────┐  │
│  │ Learning Curves  │  Predictions     │  │
│  │                  │                  │  │
│  │  [PLOT 1]        │   [PLOT 2]       │  │
│  │                  │                  │  │
│  │                  │                  │  │
│  └──────────────────┴──────────────────┘  │
│                                            │
│  REZULTATE TEST SET                        │
│  • R² = [X.XXXX]                          │
│  • RMSE = [X,XXX.XX]                      │
│  • MAE = [X,XXX.XX]                       │
│                                            │
│  Convergență: ~[XX] iterații              │
└────────────────────────────────────────────┘
```

### Imagini de inclus:
- **Stânga**: `xgb_learning_curves.png`
- **Dreapta**: `xgb_predictions.png` (Test Set)

---

## Slide 6: Support Vector Regression 🎯

### Layout:
```
┌────────────────────────────────────────────┐
│  Model 3: Support Vector Regression (SVR) │
│                                            │
│  CONFIGURAȚIE                              │
│  • Kernel: RBF (Radial Basis Function)    │
│  • C: 100, Gamma: scale                    │
│  • Margin-based learning                   │
│                                            │
│  ┌──────────────────┬──────────────────┐  │
│  │ Kernel Compare   │  Predictions     │  │
│  │                  │                  │  │
│  │  [PLOT 1]        │   [PLOT 2]       │  │
│  │                  │                  │  │
│  │                  │                  │  │
│  └──────────────────┴──────────────────┘  │
│                                            │
│  REZULTATE TEST SET                        │
│  • R² = [X.XXXX]                          │
│  • RMSE = [X,XXX.XX]                      │
│  • MAE = [X,XXX.XX]                       │
│                                            │
│  Support Vectors: [XX]% din training      │
└────────────────────────────────────────────┘
```

### Imagini de inclus:
- **Stânga**: `svr_kernel_comparison.png`
- **Dreapta**: `svr_predictions.png` (Test Set)

---

## Slide 7: Neural Network 🧠

### Layout:
```
┌────────────────────────────────────────────┐
│  Model 4: Neural Network (MLP Regressor)  │
│                                            │
│  ARHITECTURĂ                               │
│  Input → [100] → [50] → [30] → Output     │
│  • Activation: ReLU                        │
│  • Optimizer: Adam                         │
│  • Regularization: L2 + Early Stopping    │
│                                            │
│  ┌──────────────────┬──────────────────┐  │
│  │ Learning Curves  │  Predictions     │  │
│  │                  │                  │  │
│  │  [PLOT 1]        │   [PLOT 2]       │  │
│  │                  │                  │  │
│  │                  │                  │  │
│  └──────────────────┴──────────────────┘  │
│                                            │
│  REZULTATE TEST SET                        │
│  • R² = [X.XXXX]                          │
│  • RMSE = [X,XXX.XX]                      │
│  • MAE = [X,XXX.XX]                       │
│                                            │
│  Convergență: [XXX] iterații              │
└────────────────────────────────────────────┘
```

### Imagini de inclus:
- **Stânga**: `nn_learning_curves.png`
- **Dreapta**: `nn_predictions.png` (Test Set)

---

## Slide 8: Comparație & Analiză Statistică 📊

### Layout:
```
┌────────────────────────────────────────────┐
│  Comparație Modele - Analiză Statistică   │
│                                            │
│  ┌──────────────────┬──────────────────┐  │
│  │  Radar Chart     │ Wilcoxon p-vals  │  │
│  │                  │                  │  │
│  │  [PLOT 1]        │   [PLOT 2]       │  │
│  │                  │                  │  │
│  └──────────────────┴──────────────────┘  │
│                                            │
│  RANKING (R² Score)                        │
│  ┌────────────────────────────────────┐   │
│  │ 🥇 [Model 1]: R² = X.XXXX         │   │
│  │ 🥈 [Model 2]: R² = X.XXXX         │   │
│  │ 🥉 [Model 3]: R² = X.XXXX         │   │
│  │  4. [Model 4]: R² = X.XXXX         │   │
│  └────────────────────────────────────┘   │
│                                            │
│  WILCOXON TEST                             │
│  • [M1] vs [M2]: p=[X.XX] → Semnificativ │
│  • [M1] vs [M3]: p=[X.XX] → Nu            │
│                                            │
│  Cross-Validation: 30 runs, mean ± std    │
└────────────────────────────────────────────┘
```

### Imagini de inclus:
- **Stânga**: `comparison_radar_chart.png`
- **Dreapta**: `wilcoxon_pvalues_heatmap.png`

### Tabel Ranking:
Folosește emoji sau iconițe colorate pentru clasament

---

## Slide 9: Integrare AI & Clustering 🤖

### Layout:
```
┌────────────────────────────────────────────┐
│  Analiză Avansată: AI & Live Data         │
│                                            │
│  🧠 GOOGLE GEMINI ("Senior Analyst")       │
│  • Rapoarte generate automat               │
│  • Context istoric și analiză brand        │
│                                            │
│  🧩 CLUSTERING (Unsupervised Learning)     │
│  • K-Means: Segmentare automată (4 grupe)  │
│  • [Mini plot: Clustering Scatter]         │
│                                            │
│  🌐 LIVE DATA (Yahoo Finance)              │
│  • Preț acțiuni în timp real               │
│  • Impact financiar curent                 │
│                                            │
│  Exemplu: "BMW scade cu 2% azi, risc mediu"│
└────────────────────────────────────────────┘
```

### Elemente Cheie:
- Subliniază cuvintele cheie: **AI**, **Clustering**, **Live**
- Arată că proiectul e mai mult decât o simplă predicție
- Poți pune screenshot din tab-ul "Brand Encyclopedia" sau "Data Explorer"

---

## Slide 10: Concluzii 🎓

### Layout:
```
┌────────────────────────────────────────────┐
│  Concluzii și Perspective                 │
│                                            │
│  🏆 MODEL CÂȘTIGĂTOR                      │
│  ┌────────────────────────────────────┐   │
│  │  [Model Name]                      │   │
│  │  R² = X.XXXX  |  RMSE = X,XXX.XX  │   │
│  └────────────────────────────────────┘   │
│                                            │
│  🔑 FACTORI CHEIE PREDICTORI              │
│  • Engine Size (dimensiune motor)          │
│  • Curb Weight (greutate vehicul)          │
│  • Horsepower (putere motor)               │
│  • Make (marca vehiculului)                │
│                                            │
│  ✅ APLICABILITATE PRACTICĂ               │
│  • Evaluare automată prețuri automobile   │
│  • Platforme de vânzare second-hand       │
│  • Sistem decizie credit auto              │
│                                            │
│  🚀 ÎMBUNĂTĂȚIRI VIITOARE                 │
│  • Ensemble voting între top 3            │
│  • Dataset mai mare pentru generalizare   │
│  • Integrare features temporale (an)      │
│                                            │
│            Vă mulțumesc!                   │
│             Întrebări?                     │
└────────────────────────────────────────────┘
```

### Design Special:
- Fundal colorat sau gradient pentru secțiunea "Model Câștigător"
- Iconițe pentru fiecare secțiune
- Text "Vă mulțumesc!" mare și centrat

---

## 🎨 Tips Design PowerPoint

### Fonts:
```
Titluri slide:        Arial Bold, 32pt
Subtitluri:          Arial Bold, 24pt
Bullet points:       Arial Regular, 20pt
Captions imagini:    Arial Italic, 16pt
```

### Culori Recomandate:

**Variantă 1 - Profesional Albastru:**
```
Background:      #FFFFFF (alb)
Text Principal:  #1F2937 (gri închis)
Accent 1:        #3B82F6 (albastru)
Accent 2:        #F59E0B (portocaliu)
Accent 3:        #10B981 (verde)
```

**Variantă 2 - Modern Închis:**
```
Background:      #1E293B (albastru închis)
Text Principal:  #F1F5F9 (alb-gri)
Accent 1:        #60A5FA (albastru deschis)
Accent 2:        #FBBF24 (galben-auriu)
Accent 3:        #34D399 (verde mint)
```

### Spacing:
- Margini: minimum 1cm pe toate laturile
- Spațiu între bullet points: 1.5 line spacing
- Titlu la minim 2cm de partea de sus

---

## 📥 Cum Inserezi Imaginile

### Pentru fiecare plot:

1. **Insert → Pictures → This Device**
2. Selectează imaginea (ex: `rf_feature_importance.png`)
3. **Right-click → Size and Position**
   - Width: 15-18 cm
   - Lock aspect ratio: ✓
4. **Aranjare**:
   - Pentru 2 plots: 50% width fiecare
   - Aliniază-le uniform
5. **Adaugă Caption** (opțional):
   - Insert → Text Box
   - Sub imagine: "Fig. 1: Feature Importance - Random Forest"

---

## ✅ Checklist Final PPT

- [ ] 10 slides (Title + 9 conținut)
- [ ] Toate plot-urile inserate și alinate
- [ ] Font consistent pe toate slide-urile
- [ ] Numerele reale completate (R², RMSE, etc.)
- [ ] Numele tău pe title slide
- [ ] Verificat ortografie
- [ ] Transitions simple (Fade, 0.5s)
- [ ] No animations pe conținut (doar slide transitions)
- [ ] Testat prezentarea în modul Slideshow
- [ ] Salvat ca .pptx ȘI .pdf (backup)

---

## 💾 Export Format

**Salvează 3 versiuni:**
1. `.pptx` - versiunea editabilă
2. `.pdf` - backup pentru prezentare
3. `.pptx` pe USB - safety backup

---

**Gata! Ai toate informațiile pentru un PowerPoint de 10/10! 🌟**