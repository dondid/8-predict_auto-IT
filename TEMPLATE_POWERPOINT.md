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

---

## Slide 2: Context & Dataset 📋

### Layout:
```
┌────────────────────────────────────────────┐
│  Dataset & Obiective                      │
│                                            │
│  DATASET UCI AUTOMOBILE (1985)             │
│  • 205 instanțe, 26 atribute (Features)    │
│  • Target: Price (Variabilă continuă)      │
│  • Key Features: horsepower, engine-size   │
│                                            │
│  OBIECTIVE PROIECT                         │
│  1. ML Predictiv: Estimare preț (Regresie) │
│  2. Măsurare Risc: Safety Score calculat   │
│  3. Consultanță: Agent AI integrat         │
│                                            │
│  "Un sistem complet: de la date brute la   │
│   decizie de investiție asistată de AI."   │
└────────────────────────────────────────────┘
```

---

## Slide 3: Arhitectura Tehnică (Hybrid) 🏗️

### Layout:
```
┌───────────────────────┬──────────────────────┐
│  CORE ML ENGINE       │  MODERN LAYERS       │
│  (Python / Scikit)    │  (API / Streamlit)   │
│                       │                      │
│  1. Preprocessing     │  3. Live Data Layer  │
│     Clean & Scale     │     Yahoo Finance    │
│        ↓              │        ↓             │
│  2. Models (4)        │  4. AI Analyst       │
│     RF, XGB, SVR, NN  │     Google Gemini    │
│        ↓              │        ↓             │
│  3. Validation        │  5. Dashboard        │
│     Wilcoxon Test     │     Streamlit UI     │
└───────────────────────┴──────────────────────┘
```
**Esențial din README**: Subliniază structura modulară: ML Clasic + AI Modern.

---

## Slide 4: ML Performance & Top Features 🏆

### Layout:
```
┌────────────────────────────────────────────┐
│  Rezultate Supervised Learning            │
│  (XGBoost vs Random Forest)                │
│                                            │
│  PERFORMANȚĂ (R² Score)                    │
│  • XGBoost: ~0.91 (Campion)                │
│  • Random Forest: ~0.89                    │
│                                            │
│  FACTORI DETERMINANȚI (Feature Importance) │
│  1. Engine-Size (Dimensiune motor)         │
│  2. Curb-Weight (Greutate)                 │
│  3. Horsepower (Cai putere)                │
│                                            │
│  [Include grafic: presentation_feature_importance.png]
└────────────────────────────────────────────┘
```

---

## Slide 5: Validare & Unsupervised Learning 🧩

### Layout:
```
┌────────────────────┬───────────────────────┐
│ Validare Statistică│ Clustering (K-Means)  │
│ (Wilcoxon Test)    │ (Bonus Feature)       │
│                    │                       │
│ • p-value < 0.05   │ • Segmentare Piață    │
│ • Diferențe reale  │ • 4 Clustere:         │
│   între modele     │   Economic, Sport,    │
│                    │   Lux, SUV            │
│                    │                       │
│ [Heatmap Image]    │ [Scatter Plot Image]  │
└────────────────────┴───────────────────────┘
```
**Esențial**: Demonstrează rigoarea academică (Wilcoxon) și inovația (Clustering).

---

## Slide 6: AI & Live Market Data 🧠

### Layout:
```
┌────────────────────────────────────────────┐
│  "Senior Analyst" - Google Gemini         │
│                                            │
│  CUM FUNCȚIONEAZĂ:                         │
│  1. Dashboard trimite date tehnice (ML)    │
│  2. API preia date financiare (Yahoo)      │
│  3. Gemini generează raport complet        │
│                                            │
│  EXEMPLU REAL:                             │
│  • Input: "BMW, 182cp, Preț ML: $25k"      │
│  • Live: "Acțiuni BMW scad cu 1.5%"        │
│  • Verdict AI: "Preț corect, dar risc      │
│    de depreciere pe termen scurt."         │
└────────────────────────────────────────────┘
```

---

## Slide 7: Dashboard & Demo 🎥

### Layout:
```
┌────────────────────────────────────────────┐
│  Interfață Utilizator (Streamlit)          │
│                                            │
│  FUNCȚIONALITĂȚI CHEIE:                    │
│  • Filtrare Avansată (Brand, Preț, Tip)    │
│  • Galerie Grafice Interactive             │
│  • Export Raport PDF/ZIP                   │
│                                            │
│  [Screenshot mare cu Dashboard-ul]         │
│                                            │
│  "Transformăm codul într-un produs finit"  │
└────────────────────────────────────────────┘
```

---

## Slide 8: Concluzii 🎓

### Layout:
```
┌────────────────────────────────────────────┐
│  Concluzii și Perspective                 │
│                                            │
│  🏆 MODEL CÂȘTIGĂTOR                      │
│  ┌────────────────────────────────────┐   │
│  │  XGBoost / Random Forest           │   │
│  │  R² = 0.9X  |  RMSE = 2XXX $       │   │
│  └────────────────────────────────────┘   │
│                                            │
│  ✅ CE AM REALIZAT                        │
│  • Pipeline ML complet & Validat          │
│  • Inovație prin AI + Live Data           │
│  • Interfață de nivel comercial           │
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

- [ ] 8 slides (Title + 7 conținut)
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