# LIBRĂRII STANDARD PYTHON
import hashlib
import logging
import os
import sqlite3
import textwrap
import warnings
from datetime import datetime
import io

# PROCESARE DATE, MATEMATICĂ ȘI STATISTICĂ
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from dotenv import load_dotenv

# MACHINE LEARNING (Scikit-Learn)
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
)

# DEEP LEARNING (PyTorch & Transformers)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# VIZUALIZARE DATE
import matplotlib.pyplot as plt
import seaborn as sns

# LOGIC PROGRAMMING & MODULE LOCALE
try:
    from pyswip import Prolog
    PROLOG_AVAILABLE = True
except (ImportError, Exception):
    PROLOG_AVAILABLE = False
    class Prolog:
        def assertz(self, x): pass
        def retractall(self, x): pass
        def query(self, x): return []
# Update import path for integration
try:
    from src.data.companii_automotive import COMPANII_AUTOMOTIVE
except ImportError:
    # Fallback if running standalone
    from companii_automotive import COMPANII_AUTOMOTIVE

# CONFIGURARE MEDIU
load_dotenv(override=True)
warnings.filterwarnings('ignore')

# Dezactivare log-uri Transformers
transformers_logger = logging.getLogger("transformers.modeling_utils")
transformers_logger.setLevel(logging.ERROR)


# UTILITĂȚI GENERALE
def hash_file(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# CONFIGURARE PRAGURI (UNIFORM PENTRU TOATE MODELELE)
class ConfigurarePraguri:
    """
    Praguri centralizate pentru interpretarea evoluției procentuale.
    """
    PRAG_CUMPARA = 1.5  # evoluție > 1.5%  => CUMPĂRĂ
    PRAG_VINDE = -1.5  # evoluție < -1.5% => VINDE

    # între -1.5% și +1.5% => PĂSTREAZĂ

    @classmethod
    def clasifica_evolutie(cls, evolutie_procent):

        if evolutie_procent >= cls.PRAG_CUMPARA:
            return 1, "CUMPĂRĂ"
        elif evolutie_procent <= cls.PRAG_VINDE:
            return 0, "VINDE"
        else:
            return 2, "PĂSTREAZĂ"


# BAZA DE DATE
class BazaDate:
    def __init__(self, db_path="sistem_financiar.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False) # check_same_thread=False for Streamlit
        self.cursor = self.conn.cursor()
        self._initializeaza_tabele()

    def _initializeaza_tabele(self):
        tabele = [
            """CREATE TABLE IF NOT EXISTS date_bursa (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simbol TEXT,
                data TEXT,
                pret_deschidere REAL,
                pret_inchidere REAL,
                maxim REAL,
                minim REAL,
                volum INTEGER,
                evolutie_procent REAL
            )""",
            """CREATE TABLE IF NOT EXISTS predictii_modele (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                simbol TEXT,
                model TEXT,
                predictie TEXT,
                confidenta REAL,
                detalii TEXT
            )"""
        ]
        for tabel in tabele:
            self.cursor.execute(tabel)
        self.conn.commit()

    def salveaza_date_bursa(self, simbol, df):

        for idx, row in df.iterrows():
            if "Open" not in row.index or "Close" not in row.index:
                continue

            data_val = idx
            if isinstance(data_val, pd.Timestamp):
                data_val = data_val.strftime("%Y-%m-%d")

            high = float(row["High"]) if "High" in row.index and not pd.isna(row["High"]) else float(row["Close"])
            low = float(row["Low"]) if "Low" in row.index and not pd.isna(row["Low"]) else float(row["Open"])
            volume = int(row["Volume"]) if "Volume" in row.index and not pd.isna(row["Volume"]) else 0

            evolutie = ((row["Close"] - row["Open"]) / row["Open"]) * 100

            self.cursor.execute(
                """
                INSERT INTO date_bursa
                (simbol, data, pret_deschidere, pret_inchidere, maxim, minim, volum, evolutie_procent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    simbol,
                    data_val,
                    float(row["Open"]),
                    float(row["Close"]),
                    high,
                    low,
                    volume,
                    float(evolutie),
                ),
            )

        self.conn.commit()

    def numara_inregistrari_pentru_simbol(self, simbol):
        self.cursor.execute("SELECT COUNT(*) FROM date_bursa WHERE simbol=?", (simbol,))
        return self.cursor.fetchone()[0]

    def obtine_date_pentru_simbol(self, simbol, zile=250):

        self.cursor.execute(
            """
            SELECT simbol, data, pret_deschidere, pret_inchidere,
                   maxim, minim, volum, evolutie_procent
            FROM date_bursa
            WHERE simbol=?
            ORDER BY data ASC
            """,
            (simbol,),
        )
        rezultate = self.cursor.fetchall()
        if not rezultate:
            return []
        return rezultate[-min(zile, len(rezultate)):]

    def salveaza_predictie(self, simbol, model, predictie, confidenta, detalii):
        self.cursor.execute(
            """
            INSERT INTO predictii_modele
            (timestamp, simbol, model, predictie, confidenta, detalii)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (datetime.now().isoformat(), simbol, model, str(predictie), confidenta, detalii),
        )
        self.conn.commit()


# HUGGINGFACE NER SIMPLIFICAT PENTRU COMPANII
class HuggingFaceNER:
    def __init__(self, companii):
        # companii: listă de tuple (nume, simbol, țară)
        self.simboluri = {nume.lower(): simbol for nume, simbol, tara in companii}
        # simplificăm: nu forțăm pipeline NER real; fallback este mapping-ul
        self.ner = None

    def extrage_companie(self, text):
        text_lower = text.lower()
        for companie, simbol in self.simboluri.items():
            if companie in text_lower:
                return companie.title(), simbol
        # Fallback default or search logic could be improved
        return "Tesla", "TSLA"


# YAHOO FINANCE
class YahooFinance:
    def obtine_date_bursa(self, simbol, perioada="5d"):

        try:
            ticker = yf.Ticker(simbol)
            istoric = ticker.history(period=perioada)
            if len(istoric) < 2:
                return None

            return {
                "pret_ieri": float(istoric["Close"].iloc[-2]),
                "pret_azi": float(istoric["Close"].iloc[-1]),
                "volum": int(istoric["Volume"].iloc[-1]),
                "istoric_complet": istoric,
            }
        except Exception as e:
            print(f"Eroare Yahoo Finance pentru {simbol}: {e}")
            return None

    def descarca_istoric_10_ani(self, simbol):

        try:
            df = yf.download(simbol, period="10y", interval="1d", auto_adjust=False, progress=False)
            
            # Fix for yfinance >= 0.2 returning MultiIndex columns
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    # Keep only the top level (Price Type) and drop the Ticker level
                    df.columns = df.columns.get_level_values(0)
                
                # Ensure columns are clean
                df = df.dropna(how='all')
                
            return df if df is not None and not df.empty else None
        except Exception as e:
            print(f"Eroare descărcare istoric pentru {simbol}: {e}")
            return None


# DATA MANAGER (VERIFICĂ ȘI DESCARCĂ DATE)
class DataManager:
    def __init__(self, db: BazaDate, hf: HuggingFaceNER, yf: YahooFinance):
        self.db = db
        self.hf = hf
        self.yf = yf

    def verifica_si_descarca_date(self, prag_min_randuri=200):
        print("\nVerificare date istorice în baza de date...")

        simboluri = list(self.hf.simboluri.values())
        trebuie_descarcate = False

        for simbol in simboluri:
            count = self.db.numara_inregistrari_pentru_simbol(simbol)
            if count < prag_min_randuri:
                print(f"Simbol {simbol} are doar {count} rânduri → trebuie descărcat.")
                trebuie_descarcate = True
                break

        if trebuie_descarcate:
            print("\nDescărcare date istorice pentru toate companiile...")
            for simbol in simboluri:
                df = self.yf.descarca_istoric_10_ani(simbol)
                if df is not None and not df.empty:
                    self.db.salveaza_date_bursa(simbol, df)
                    print(f"✓ Descărcat {len(df)} rânduri pentru {simbol}")
            print("Descărcare completă.")
        else:
            print("Baza de date conține suficiente date istorice.")

    def asigura_date_pentru_simbol(self, simbol, prag_min=200):
        count = self.db.numara_inregistrari_pentru_simbol(simbol)
        if count < prag_min:
            print(f"Date lipsă pentru {simbol} ({count} < {prag_min}). Descărcare...")
            df = self.yf.descarca_istoric_10_ani(simbol)
            if df is not None and not df.empty:
                self.db.salveaza_date_bursa(simbol, df)
                return True
            return False
        return True


# EXPORT CSV CU VALIDARE CRONOLOGICĂ
class ExportDate:
    def __init__(self, db: BazaDate, output_dir="date_per_simbol"):
        self.db = db
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def exporta(self, simbol, zile=250):
        inregistrari = self.db.obtine_date_pentru_simbol(simbol, zile)
        if not inregistrari:
            return None, None
        df = pd.DataFrame(
            inregistrari,
            columns=["Simbol", "Data", "Open", "Close", "High", "Low", "Volume", "Evolutie_%"],
        )
        df = df.sort_values("Data")
        filename = f"{self.output_dir}/{simbol}_{zile}_zile.csv"
        # We don't necessarily need to save to disk for the web app, but we can
        try:
            df.to_csv(filename, index=False)
            filehash = hash_file(filename)
        except Exception:
            filehash = "memory"
        return df, filehash


# ANALIZA NUMPY (METRICI FINANCIARE)
class AnalizaNumPy:
    @staticmethod
    def calculeaza_evolutie(pret_ieri, pret_azi):
        return ((pret_azi - pret_ieri) / pret_ieri) * 100

    def calculeaza_metrici(self, date_bursa):

        if not date_bursa:
            return None

        pret_ieri = date_bursa["pret_ieri"]
        pret_azi = date_bursa["pret_azi"]
        evolutie = self.calculeaza_evolutie(pret_ieri, pret_azi)

        if "istoric_complet" in date_bursa:
            istoric = date_bursa["istoric_complet"]["Close"].values
            if len(istoric) > 0:
                volatilitate = np.std(istoric) / np.mean(istoric) * 100
            else:
                volatilitate = 0
        else:
            volatilitate = 0

        trend = "crescator" if evolutie > 0 else "descrescator" if evolutie < 0 else "stabil"

        preturi = np.array([pret_ieri, pret_azi])
        if np.std(preturi) > 0:
            momentum = (pret_azi - np.mean(preturi)) / np.std(preturi)
        else:
            momentum = 0

        return {
            "evolutie_procent": float(evolutie),
            "volatilitate": float(volatilitate),
            "trend": trend,
            "momentum": float(momentum),
        }


# ANALIZA PROLOG (RAȚIONAMENT SIMBOLIC)
class AnalizaProlog:
    def __init__(self):
        self.available = PROLOG_AVAILABLE
        if self.available:
            try:
                self.prolog = Prolog()
                self._init_ontologie()
            except Exception as e:
                print(f"Prolog initialization failed: {e}")
                self.available = False

    def _init_ontologie(self):
        reguli = [
            # --- LOGICĂ DE TREND ---
            "trend(crestere) :- evolutie(mare)",
            "trend(scadere) :- evolutie(mica)",
            "trend(stabil) :- evolutie(moderata)",

            # --- LOGICĂ DE RISC ---
            "risc(scazut) :- volatilitate(scazuta)",
            "risc(moderat) :- volatilitate(moderata)",
            "risc(ridicat) :- volatilitate(mare)",

            # --- MATRICEA DE DECIZIE (Acoperire 100%) ---
            # Cazuri CUMPĂRĂ
            "decizie(cumpara) :- trend(crestere), risc(scazut)",

            # Cazuri PĂSTREAZĂ
            "decizie(pastreaza) :- trend(crestere), risc(moderat)",
            "decizie(pastreaza) :- trend(stabil)",
            "decizie(pastreaza) :- trend(crestere), risc(ridicat)",  # Prea riscant pentru buy

            # Cazuri VINDE
            "decizie(vinde) :- trend(scadere)",  # Vindem pe scădere indiferent de risc (conservator)

            # --- EXPLICAȚII ---
            "explicatie(cumpara, 'Semnal puternic: crestere cu risc minim')",
            "explicatie(pastreaza, 'Pozitie neutra: piata stabila sau risc prea mare pentru tranzactionare')",
            "explicatie(vinde, 'Semnal de vinde: trend descendent confirmat')"
        ]

        for r in reguli:
            try:
                self.prolog.assertz(r)
            except Exception as e:
                print(f"Eroare Prolog Syntax: {e}")

    def adauga_fapte(self, metrici):
        if not self.available: return
        try:
            self.prolog.retractall("evolutie(_)")
            self.prolog.retractall("volatilitate(_)")

            e = metrici["evolutie_procent"]
            if e >= ConfigurarePraguri.PRAG_CUMPARA:
                self.prolog.assertz("evolutie(mare)")
            elif e <= ConfigurarePraguri.PRAG_VINDE:
                self.prolog.assertz("evolutie(mica)")
            else:
                self.prolog.assertz("evolutie(moderata)")

            v = metrici["volatilitate"]
            if v > 8.0:
                self.prolog.assertz("volatilitate(mare)")
            elif v > 3.0:
                self.prolog.assertz("volatilitate(moderata)")
            else:
                self.prolog.assertz("volatilitate(scazuta)")
        except Exception:
            pass

    def rationeaza(self):
        if not self.available:
            return "necunoscut", "Prolog indisponibil"
            
        try:
            decizie = list(self.prolog.query("decizie(X)"))
            if not decizie:
                return "necunoscut", "Fara reguli aplicabile"

            d = decizie[0]["X"]
            explic = list(self.prolog.query(f"explicatie({d}, E)"))
            return d, explic[0]["E"] if explic else "Fara explicatie"
        except Exception:
            return "necunoscut", "Eroare executie Prolog"


# DATASET BUILDERS PENTRU NEXT-DAY PREDICTION
class MLDatasetBuilder_Tabular:

    def __init__(self, n_lags=5):
        self.n_lags = n_lags
        self.scaler = StandardScaler()

    def construieste_dataset(self, df):
        df = df.sort_values("Data").copy()

        df["Target"] = df["Evolutie_%"].shift(-1)

        for lag in range(1, self.n_lags + 1):
            df[f"Close_lag_{lag}"] = df["Close"].shift(lag)
            df[f"Volume_lag_{lag}"] = df["Volume"].shift(lag)

        df = df.dropna()

        features = [c for c in df.columns if "lag" in c]
        X = df[features].values
        y_raw = df["Target"].values

        y_class = np.where(y_raw > 0.5, 1, np.where(y_raw < -0.5, -1, 0))

        if len(X) > 0:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
            
        return X_scaled, y_class


class MLDatasetBuilder_Sequence:

    def __init__(self, window=30):
        self.window = window
        self.scaler = StandardScaler()

    def construieste_dataset(self, df):
        df = df.sort_values("Data").copy()
        features = ["Open", "High", "Low", "Close", "Volume"]
        data = df[features].values
        
        if len(data) == 0:
            return np.array([]), np.array([])
            
        data_scaled = self.scaler.fit_transform(data)

        target_raw = df["Evolutie_%"].shift(-1).values

        X_seq, y_seq = [], []
        for i in range(len(df) - self.window - 1):
            X_seq.append(data_scaled[i: i + self.window])
            y_seq.append(target_raw[i + self.window])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        y_class = np.where(y_seq > 0.5, 1, np.where(y_seq < -0.5, -1, 0))

        return X_seq, y_class


# MODELE DATE TABELARE
class RFModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=300, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)


class NNModel:
    def __init__(self, input_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        ).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, X_train, y_train, epochs=30, batch_size=32):
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_train + 1, dtype=torch.long).to(self.device)  # -1->0,0->1,1->2

        dataset = TensorDataset(X_tensor, y_tensor)
        if len(dataset) > 0:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
            self.model.train()
            for _ in range(epochs):
                for xb, yb in loader:
                    self.optimizer.zero_grad()
                    loss = self.criterion(self.model(xb), yb)
                    loss.backward()
                    self.optimizer.step()

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            preds = torch.argmax(outputs, axis=1).cpu().numpy() - 1  # 0->-1,1->0,2->1
        return preds


class NaiveBayesTabular:
    def __init__(self):
        self.model = GaussianNB()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class MarkovChainModel:

    def __init__(self):
        self.transition_matrix = None

    def train(self, y):
        mapping = {-1: 0, 0: 1, 1: 2}
        y_idx = np.array([mapping[v] for v in y])
        matrix = np.zeros((3, 3))
        for i in range(len(y_idx) - 1):
            matrix[y_idx[i], y_idx[i + 1]] += 1

        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.transition_matrix = matrix / row_sums

    def predict(self, last_class):
        mapping = {-1: 0, 0: 1, 1: 2}
        inv_mapping = {0: -1, 1: 0, 2: 1}
        idx = mapping[last_class]
        probs = self.transition_matrix[idx]
        pred_idx = np.argmax(probs)
        return inv_mapping[pred_idx]


# MODELE SECVENȚIALE
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class TCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.tcn1 = TCNBlock(input_dim, hidden_dim, dilation=1)
        self.tcn2 = TCNBlock(hidden_dim, hidden_dim, dilation=2)
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = x[:, :, -1]
        return self.fc(x)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 3)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.fc(x)


# 9GPT-2
class GPT2Model:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def genereaza(self, context, max_length=290):

        inputs = self.tokenizer(context, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.8,
                top_p=0.92,
                top_k=50,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def genereaza_analiza_nlp(self, simbol, metrici):
        context = (
            f"Analysis for {simbol}: "
            f"The stock price change is {metrici['evolutie_procent']:.2f}%. "
            f"The current trend is {metrici['trend']} and volatility is {metrici['volatilitate']:.2f}. "
            f"In my opinion, this means that"
        )

        insight = self.genereaza(context, max_length=100)
        return insight.replace(context, "").strip()


# GROK AI
class GrokModel:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GROK_API_KEY")

    def genereaza(self, context, system_instruction=None):
        """Generează răspuns folosind Grok AI. 
           system_instruction: Prompt de sistem opțional (ex: 'Ești un expert auto'). 
           Dacă e None, folosește default-ul financiar."""
        if not self.api_key:
            return "GROK API KEY lipsă – folosesc modul simulat (Grok este offline)."

        if system_instruction is None:
            system_instruction = "Ești un asistent financiar profesionist. Răspunzi STRICT pe baza datelor furnizate."

        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {
                        "role": "system",
                        "content": system_instruction
                    },
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                "temperature": 0.5, # Puțin mai creativ pentru chat
                "max_tokens": 800
            }

            response = requests.post(url, headers=headers, json=payload, timeout=30)
            data = response.json()

            if "error" in data:
                return f"Eroare GROK API: {data['error']['message']}"

            return data["choices"][0]["message"]["content"]

        except Exception as e:
            return f"Eroare GROK API: {str(e)}"


# PIPELINE UNIFICAT NEXT-DAY PREDICTOR (INCLUZÂND NUMPY + PROLOG)
class NextDayPredictor:
    def __init__(self, window=30, n_lags=5):
        self.window = window
        self.n_lags = n_lags

        self.tab_builder = MLDatasetBuilder_Tabular(n_lags=n_lags)
        self.seq_builder = MLDatasetBuilder_Sequence(window=window)

        self.numpy_analyzer = AnalizaNumPy()
        self.prolog_analyzer = AnalizaProlog()

    def _train_and_predict_seq(self, model, X_train, y_train, X_last):
        device = next(model.parameters()).device

        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train + 1, dtype=torch.long).to(device)

        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(15):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        model.eval()
        X_last_t = torch.tensor(X_last.reshape(1, X_last.shape[0], X_last.shape[1]), dtype=torch.float32).to(device)

        with torch.no_grad():
            out = model(X_last_t)
            pred = torch.argmax(out, axis=1).cpu().numpy()[0] - 1

        return pred

    def ruleaza(self, df):
        rezultate = {}

        # 1. Pregătire date
        X_tab, y_tab = self.tab_builder.construieste_dataset(df)
        X_seq, y_seq = self.seq_builder.construieste_dataset(df)

        # Verificare minimală date
        if len(X_tab) < 10 or len(X_seq) < 10:
            print("Prea puține date pentru modelele ML.")
            return {}

        # Split date pentru ML
        split_t = int(len(X_tab) * 0.8)
        X_train_tab, y_train_tab = X_tab[:split_t], y_tab[:split_t]

        split_s = int(len(X_seq) * 0.8)
        X_train_seq, y_train_seq = X_seq[:split_s], y_seq[:split_s]
        X_test_seq_last = X_seq[-1]

        # 1763 ─ NaiveBayes
        nb = NaiveBayesTabular()
        nb.train(X_train_tab, y_train_tab)
        rezultate["NaiveBayes"] = nb.predict(X_tab[-1].reshape(1, -1))[0]

        # 1906 ─ MarkovChain
        mc = MarkovChainModel()
        mc.train(y_train_tab)
        rezultate["MarkovChain"] = mc.predict(y_train_tab[-1])

        # 1972 ─ Prolog
        try:
            pret_azi = df["Close"].iloc[-1]
            pret_ieri = df["Close"].iloc[-2]
            metrici = self.numpy_analyzer.calculeaza_metrici({
                "pret_ieri": pret_ieri,
                "pret_azi": pret_azi,
                "istoric_complet": df.set_index("Data")
            })
            self.prolog_analyzer.adauga_fapte(metrici)
            decizie_p, explicatie = self.prolog_analyzer.rationeaza()
            mapping = {"vinde": -1, "pastreaza": 0, "cumpara": 1, "necunoscut": None}
            # For consistency, return INT code
            code = mapping.get(decizie_p, 0)
            rezultate["Prolog"] = code
            rezultate["Prolog_explicatie"] = explicatie
        except:
            rezultate["Prolog"] = 0

        # 1986 ─ NeuralNetwork (Backpropagation)
        nn_m = NNModel(input_dim=X_tab.shape[1])
        nn_m.train(X_train_tab, y_train_tab)
        rezultate["NeuralNetwork"] = nn_m.predict(X_tab[-1].reshape(1, -1))[0]

        # 1997 ─ LSTM
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lstm = LSTMModel(input_dim=X_seq.shape[2]).to(device)
        rezultate["LSTM"] = self._train_and_predict_seq(lstm, X_train_seq, y_train_seq, X_test_seq_last)

        # 2001 ─ RandomForest
        rf = RFModel()
        rf.train(X_train_tab, y_train_tab)
        rezultate["RandomForest"] = rf.predict(X_tab[-1].reshape(1, -1))[0]

        # 2014 ─ GRU
        gru = GRUModel(input_dim=X_seq.shape[2]).to(device)
        rezultate["GRU"] = self._train_and_predict_seq(gru, X_train_seq, y_train_seq, X_test_seq_last)

        # 2016 ─ TCN
        tcn = TCNModel(input_dim=X_seq.shape[2]).to(device)
        rezultate["TCN"] = self._train_and_predict_seq(tcn, X_train_seq, y_train_seq, X_test_seq_last)

        # 2017 ─ Transformer
        trans = TransformerModel(input_dim=X_seq.shape[2]).to(device)
        rezultate["Transformer"] = self._train_and_predict_seq(trans, X_train_seq, y_train_seq, X_test_seq_last)

        return rezultate


#  CALCULATOR METRICI DE PERFORMANȚĂ
class PerformanceMetrics:

    @staticmethod
    def calculeaza_metrici_complete(y_true, y_pred, model_name="Model"):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if len(y_true) == 0:
            return {}

        metrici = {}
        metrici['accuracy'] = accuracy_score(y_true, y_pred)
        metrici['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrici['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrici['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        metrici['directional_accuracy'] = np.mean(np.sign(y_true) == np.sign(y_pred))
        return metrici


class FinancialCoordinator:
    """Class that orchestrates the financial analysis for use in the dashboard."""
    def __init__(self):
        self.db = BazaDate()
        self.ner = HuggingFaceNER(COMPANII_AUTOMOTIVE)
        self.yf_client = YahooFinance()
        self.data_manager = DataManager(self.db, self.ner, self.yf_client)
        self.exporter = ExportDate(self.db)
        self.predictor = NextDayPredictor(window=30, n_lags=5)
        self.analizator_numpy = AnalizaNumPy()
        self.label_mapping = {-1: "VINDE", 0: "PASTREAZA", 1: "CUMPARA"}
        
    def ensure_data_updated(self):
        self.data_manager.verifica_si_descarca_date(prag_min_randuri=200)
        
    def analyze_company(self, company_name_or_symbol, window=900):
        # 1. Identify
        nume_companie, simbol = self.ner.extrage_companie(company_name_or_symbol)
        if company_name_or_symbol.upper() in [v for k,v in self.ner.simboluri.items()]:
             # If passed a symbol directly, reuse it
             simbol = company_name_or_symbol.upper()
             nume_companie = company_name_or_symbol # Approximation
        
        # 2. Get Data
        df, filehash = self.exporter.exporta(simbol, zile=window)
        if df is None or df.empty:
            # Attempt auto-download
            success = self.data_manager.asigura_date_pentru_simbol(simbol)
            if success:
                 df, filehash = self.exporter.exporta(simbol, zile=window)
            
            if df is None or df.empty:
                return {"error": f"No data found for {simbol}. Please check your internet connection or try 'Actualizează Date Istorice' in sidebar."}
            
        # 3. Metrics
        pret_azi = float(df["Close"].iloc[-1])
        pret_ieri = float(df["Close"].iloc[-2])
        metrici = self.analizator_numpy.calculeaza_metrici({
            "pret_ieri": pret_ieri,
            "pret_azi": pret_azi,
            "istoric_complet": df
        })
        
        # 4. Predictions
        rezultate = self.predictor.ruleaza(df)
        
        # 5. Consensus
        votes = [int(v) for k,v in rezultate.items() if isinstance(v, (int, np.integer)) and k != "Prolog_explicatie"]
        if votes:
            consensus = max(set(votes), key=votes.count)
        else:
            consensus = 0
            
        vote_counts = {
            "CUMPARA": votes.count(1),
            "VINDE": votes.count(-1),
            "PASTREAZA": votes.count(0)
        }

        # 6. Advanced NLP (Optional lazy load or call if needed)
        # We return the objects needed for the UI to call them if user wants
        return {
            "success": True,
            "symbol": simbol,
            "company_name": nume_companie,
            "data": df,
            "metrics": metrici,
            "predictions": rezultate,
            "consensus": self.label_mapping.get(consensus, "UNKNOWN"),
            "vote_counts": vote_counts,
            "current_price": pret_azi
        }
        
    def generate_gpt_analysis(self, simbol, metrici):
        gpt = GPT2Model()
        return gpt.genereaza_analiza_nlp(simbol, metrici)
        
    def generate_grok_analysis(self, simbol, metrici, consensus_str):
        grok = GrokModel()
         # Pregătim contextul
        context_grok = f"Analizează simbolul {simbol}. Evoluție: {metrici['evolutie_procent']}%, Trend: {metrici['trend']}. Decizie consensus: {consensus_str}."
        return grok.genereaza(context_grok)

