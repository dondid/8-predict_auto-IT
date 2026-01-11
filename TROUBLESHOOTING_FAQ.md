# 🔧 Troubleshooting & FAQ

## 🚨 Probleme Comune și Soluții

---

### ❌ Eroare 1: "ModuleNotFoundError: No module named 'X'"

**Descriere**: Python nu găsește o librărie necesară

**Cauză**: Librăriile nu sunt instalate

**Soluție**:
```bash
# Instalează toate librăriile dintr-o dată
pip install -r requirements.txt

# SAU instalează individual
pip install pandas numpy scikit-learn xgboost matplotlib seaborn shap scipy
```

**Verificare**:
```bash
python test_01_check_installation.py
```

---

### ❌ Eroare 2: "FileNotFoundError: [Errno 2] No such file or directory: 'processed_data.pkl'"

**Descriere**: Scripturile caută fișiere generate de etapele anterioare

**Cauză**: Nu ai rulat modulele în ordine

**Soluție 1** (Recomandată):
```bash
# Rulează master pipeline care face totul automat
python 00_master_pipeline.py
```

**Soluție 2** (Manual):
```bash
# Rulează în ordine
python 01_data_loading.py
python 02_data_preprocessing.py
python 03_random_forest_model.py
# etc.
```

**Verificare**:
```bash
# Verifică dacă fișierele există
ls *.pkl
```

---

### ❌ Eroare 3: "HTTPError: HTTP Error 404: Not Found" (la încărcarea datelor)

**Descriere**: Dataset-ul nu poate fi descărcat de la UCI

**Cauză**: Probleme de conexiune sau URL-ul s-a schimbat

**Soluție 1** - Descarcă manual:
1. Accesează: https://archive.ics.uci.edu/ml/datasets/automobile
2. Descarcă `imports-85.data`
3. Salvează în folder-ul proiectului
4. Modifică în `01_data_loading.py`:
```python
# În loc de URL
# df = pd.read_csv(url, names=column_names, na_values='?')

# Folosește fișierul local
df = pd.read_csv('imports-85.data', names=column_names, na_values='?')
```

**Soluție 2** - Verifică conexiunea:
```bash
ping archive.ics.uci.edu
```

---

### ❌ Eroare 4: SVR foarte lent / se blochează

**Descriere**: SVR durează foarte mult să se antreneze

**Cauză**: SVR are complexitate O(n²) - O(n³), normal pentru acest algoritm

**Soluție 1** - Reduce cross-validation:
În `05_svr_model.py`, linia ~190:
```python
# În loc de 30
cv_results = perform_cross_validation(X_combined, y_combined, n_runs=10)  # Reduce la 10
```

**Soluție 2** - Skip hyperparameter tuning:
În `05_svr_model.py`, linia ~70:
```python
svr_model = train_svr(
    data_dict['X_train'], data_dict['y_train'],
    data_dict['X_val'], data_dict['y_val'],
    tune_hyperparams=False  # Setează False
)
```

**Soluție 3** - Reduce sample size pentru CV:
```python
# Sample doar 75% din date pentru CV la SVR
X_sample = X_combined.sample(frac=0.75, random_state=42)
y_sample = y_combined[X_sample.index]
```

**Estimare timp**:
- Cu hyperparameter tuning: 20-40 minute
- Fără tuning: 5-10 minute
- Cu n_runs=10: 3-5 minute

---

### ❌ Eroare 5: "MemoryError" sau Python crashes

**Descriere**: Python consumă prea multă memorie

**Cauză**: SHAP values sau cross-validation pe date mari

**Soluție 1** - Reduce samples pentru SHAP:
În `03_random_forest_model.py`, linia ~270:
```python
shap_values, explainer = compute_shap_values(
    rf_model,
    data_dict['X_train'],
    data_dict['X_test'],
    max_samples=50  # Reduce de la 100 la 50
)
```

**Soluție 2** - Reduce cross-validation runs:
```python
cv_results = perform_cross_validation(..., n_runs=10)  # În loc de 30
```

**Soluție 3** - Închide alte aplicații:
- Browser-e
- IDE-uri grele
- Aplicații în background

**Verificare memorie disponibilă**:
```python
import psutil
print(f"RAM disponibil: {psutil.virtual_memory().available / (1024**3):.2f} GB")
```

---

### ❌ Eroare 6: "Convergence Warning" la Neural Network

**Descriere**: 
```
ConvergenceWarning: Stochastic Optimizer: Maximum iterations (500) reached and the optimization hasn't converged yet.
```

**Cauză**: Neural Network nu a avut destule iterații să ajungă la convergență

**Soluție** - Crește max_iter:
În `06_neural_network_model.py`:
```python
mlp_model = MLPRegressor(
    hidden_layer_sizes=(100, 50, 30),
    activation='relu',
    solver='adam',
    alpha=0.001,
    learning_rate='adaptive',
    max_iter=1000,  # Crește de la 500 la 1000
    early_stopping=True,
    validation_fraction=0.15,
    random_state=42,
    verbose=False
)
```

**Nu e o problemă gravă**: Modelul va funcționa, doar că ar putea avea performanță ușor mai slabă.

---

### ❌ Eroare 7: Plot-urile nu se salvează / nu apar

**Descriere**: Fișierele .png nu sunt generate

**Cauză**: Matplotlib backend sau permisiuni folder

**Soluție 1** - Setează backend explicit:
La începutul fiecărui script cu plots:
```python
import matplotlib
matplotlib.use('Agg')  # Backend non-interactive
import matplotlib.pyplot as plt
```

**Soluție 2** - Verifică permisiuni:
```bash
# Windows
icacls . /grant Users:F

# Linux/Mac
chmod 755 .
```

**Soluție 3** - Specifică path absolut:
```python
import os
save_path = os.path.join(os.getcwd(), 'rf_predictions.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
```

---

### ❌ Eroare 8: "ImportError: cannot import name 'xxx' from 'sklearn'"

**Descriere**: Funcții sklearn nu pot fi importate

**Cauză**: Versiune veche de scikit-learn

**Soluție** - Update sklearn:
```bash
pip install --upgrade scikit-learn
```

**Verificare versiune**:
```python
import sklearn
print(sklearn.__version__)  # Trebuie >= 1.2.0
```

---

### ❌ Eroare 9: Cross-validation durează foarte mult

**Descriere**: 30 de runs × 4 modele = foarte mult timp

**Cauză**: Normal - asta e partea care durează cel mai mult

**Soluție 1** - Reduce runs:
```python
# În loc de 30, folosește 10
cv_results = perform_cross_validation(..., n_runs=10)
```

**Soluție 2** - Paralelizare (advanced):
```python
from joblib import Parallel, delayed

def single_cv_run(X, y, model, seed):
    # ... logica pentru un run
    return mse, rmse, r2

results = Parallel(n_jobs=-1)(
    delayed(single_cv_run)(X, y, model, i) 
    for i in range(30)
)
```

**Estimare timp totală**:
- Quick test (reduced settings): 2-3 minute
- Pipeline complet cu tune_hyperparams=False: 15-20 minute
- Pipeline complet cu tune_hyperparams=True: 30-45 minute

---

### ❌ Eroare 10: "ValueError: could not convert string to float"

**Descriere**: Date nu pot fi convertite la numeric

**Cauză**: Encoding incomplet pentru variabile categoriale

**Verificare**:
```python
# Verifică tipurile de date
print(X_train.dtypes)
print(X_train.select_dtypes(include=['object']).columns)
```

**Soluție** - Asigură-te că toate coloanele sunt numerice după encoding:
În `02_data_preprocessing.py`, adaugă:
```python
# După encoding
print("Columns still object type:", X.select_dtypes(include=['object']).columns.tolist())

# Forțează conversie
for col in X.select_dtypes(include=['object']).columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
    X[col].fillna(X[col].median(), inplace=True)
```

---

### ❌ Eroare 11: "Streamlit command not found"

**Descriere**: Nu poți rula `streamlit run dashboard.py`

**Cauză**: `streamlit` nu e în PATH sau e instalat în alt venv

**Soluție**:
```bash
python -m streamlit run dashboard.py
```
Sau reinstalează: `pip install streamlit`

---

### ❌ Eroare 12: "Gemini API Key Missing"

**Descriere**: AI Assistant nu răspunde / apare eroare 403

**Cauză**: Cheia API nu e setată în `.env`

**Soluție**:
1. Creează fișier `.env` (copiază din `.env.example`)
2. Adaugă linia: `GEMINI_API_KEY=AIzaSy...` (cheia ta reală)
3. Restart la aplicație

---

## 🎯 Întrebări Frecvente (FAQ)

### Q1: Cât timp durează să rulez tot proiectul?

**A**: 
- **Quick test** (test_02_quick_pipeline.py): 2-3 minute
- **Pipeline complet** (fără hyperparameter tuning): 15-20 minute
- **Pipeline complet** (cu hyperparameter tuning): 30-45 minute
- **SVR cu tuning**: +20-30 minute

**Recomandare**: Rulează fără tuning pentru test, apoi cu tuning pentru rezultate finale.

---

### Q2: Care model este de obicei cel mai bun?

**A**: Pe acest dataset, de obicei:
1. **XGBoost** - 85-92% R²
2. **Random Forest** - 83-90% R²
3. **Neural Network** - 80-89% R²
4. **SVR** - 78-88% R²

Dar poate varia în funcție de split-ul aleatoriu!

---

### Q3: Pot schimba dataset-ul?

**A**: Da! Pașii:

1. **Găsește un dataset de regresie** (ex: Kaggle, UCI)
2. **Modifică în `01_data_loading.py`**:
   - URL/path către dataset
   - `column_names`
   - Numele coloanei țintă (în loc de `price`)
3. **Ajustează preprocessing în `02_data_preprocessing.py`**:
   - Logica pentru missing values
   - Feature engineering specific domeniului
4. **Rulează restul scripturilor normal**

**Sugestii dataset**:
- California Housing
- Boston Housing (similar cu automobile)
- Diamond Prices
- Insurance Costs

---

### Q4: Cum adaug un al 5-lea model?

**A**: 

1. **Creează `08_new_model.py`** copiat din unul existent
2. **Modifică modelul**:
```python
from sklearn.linear_model import Ridge  # Exemplu

def train_new_model(X_train, y_train, X_val, y_val):
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model
```
3. **Salvează rezultatele** similar cu celelalte:
```python
with open('new_model_results.pkl', 'wb') as f:
    pickle.dump(results_summary, f)
```
4. **Actualizează `07_model_comparison_statistical.py`**:
```python
model_files = {
    # ... modelele existente
    'New Model': 'new_model_results.pkl'
}
```

---

### Q5: Cum exportez rezultatele într-un format mai user-friendly?

**A**:

**Excel**:
```python
# În 07_model_comparison_statistical.py
comparison_df.to_excel('model_comparison.xlsx', index=True)
```

**HTML**:
```python
html = comparison_df.to_html()
with open('results.html', 'w') as f:
    f.write(html)
```

**LaTeX** (pentru rapoarte academice):
```python
latex = comparison_df.to_latex()
with open('results.tex', 'w') as f:
    f.write(latex)
```

---

### Q6: Pot rula pe Google Colab?

**A**: Da!

**Pași**:
1. Upload toate fișierele .py în Colab
2. Instalează dependencies:
```python
!pip install -r requirements.txt
```
3. Rulează:
```python
!python 00_master_pipeline.py
```
4. Download rezultatele:
```python
from google.colab import files

# Download toate PNG-urile
import glob
for file in glob.glob("*.png"):
    files.download(file)
```

**Avantaj**: Hardware mai puternic, GPU gratuit

---

### Q7: Cum verific că rezultatele mele sunt corecte/rezonabile?

**A**:

**Checklist validare**:
- [ ] R² între 0.7-0.95 (pentru majoritatea modelelor)
- [ ] RMSE în intervalul 2000-5000 (pentru prețuri automobile)
- [ ] MAE < RMSE (întotdeauna adevărat matematic)
- [ ] Training R² > Test R² (ușoară diferență e normală)
- [ ] Diferența Train-Test R² < 0.10 (altfel: overfitting)
- [ ] Cross-validation std rezonabilă (< 20% din mean)

**Red flags**:
- ❌ R² = 1.0 sau foarte aproape → Data leakage suspect
- ❌ R² negativ → Model mai rău decât media
- ❌ RMSE > 10,000 → Ceva e foarte greșit
- ❌ Train R² = 0.99, Test R² = 0.50 → Overfitting sever

---

### Q8: Ce fac dacă Wilcoxon test arată p > 0.05 pentru toate comparațiile?

**A**: 

**Înseamnă**: Nu există diferențe statistice semnificative între modele.

**E OK!** Poți spune:
> "Deși [Model X] are cea mai bună performanță medie (R²=0.XX), testele Wilcoxon indică că diferențele nu sunt statistic semnificative (toate p > 0.05). Aceasta sugerează că toate cele 4 modele au performanțe comparabile pe acest dataset. În practică, am alege [Model X] datorită [ușurinței interpretării / vitezei de execuție / etc.]"

**Nu e o problemă** - arată că ai analizat corect!

---

### Q9: Plot-urile nu arată bine în PowerPoint. Ce fac?

**A**:

**Soluție 1** - Crește DPI la salvare:
```python
plt.savefig('plot.png', dpi=600, bbox_inches='tight')  # Dublu față de 300
```

**Soluție 2** - Salvează ca SVG (vector):
```python
plt.savefig('plot.svg', format='svg', bbox_inches='tight')
```
Apoi în PowerPoint: Insert → Pictures → SVG

**Soluție 3** - Ajustează size în Python:
```python
fig, ax = plt.subplots(figsize=(12, 8))  # Mai mare
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
```

---

### Q10: Cum explic rezultatele la cineva non-tehnic?

**A**:

**Template simplu**:

**R² Score**:
> "R² ne spune ce procent din variația prețului e explicată de model. Un R² de 0.85 înseamnă că modelul explică 85% din diferențele de preț între mașini."

**RMSE**:
> "RMSE e eroarea medie în dolari. Un RMSE de 3000$ înseamnă că, în medie, predicțiile diferă cu ±3000$ față de prețul real."

**Wilcoxon Test**:
> "E ca un test statistic care ne spune dacă diferența între două modele e reală sau doar întâmplătoare. Dacă p < 0.05, diferența e semnificativă."

**Feature Importance**:
> "Ne arată care caracteristici ale mașinii contează cel mai mult pentru preț. De exemplu, dimensiunea motorului și puterea sunt cei mai importanți factori."

---

## 📞 Support și Resurse

### Dacă tot nu merge:

1. **Verifică log-urile**:
```bash
python 00_master_pipeline.py 2>&1 | tee pipeline_log.txt
```

2. **Postează pe forum**:
   - Stack Overflow
   - Reddit r/learnmachinelearning
   - Grupul de curs

3. **Documentație oficială**:
   - [Scikit-learn](https://scikit-learn.org/)
   - [XGBoost](https://xgboost.readthedocs.io/)
   - [Pandas](https://pandas.pydata.org/)

4. **Contactează profesorul**:
   - Email cu log-uri și screenshot-uri
   - Descrie pașii urmați

---

**Remember**: Majoritatea erorilor sunt ușor de rezolvat - nu te panica! 💪**