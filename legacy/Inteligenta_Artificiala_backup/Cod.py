# LIBRĂRII STANDARD PYTHON
import hashlib
import logging
import os
import sqlite3
import textwrap
import warnings
from datetime import datetime
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
    confusion_matrix,matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
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
from pyswip import Prolog
from companii_automotive import COMPANII_AUTOMOTIVE

# CONFIGURARE MEDIU
load_dotenv()
warnings.filterwarnings('ignore')

# Dezactivare log-uri  Transformers
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
        self.conn = sqlite3.connect(db_path)
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
            df = yf.download(simbol, period="10y", interval="1d", auto_adjust=False)
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
        df.to_csv(filename, index=False)
        filehash = hash_file(filename)
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
        self.prolog = Prolog()
        self._init_ontologie()

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

    def rationeaza(self):
        decizie = list(self.prolog.query("decizie(X)"))
        if not decizie:
            return "necunoscut", "Fara reguli aplicabile"

        d = decizie[0]["X"]
        explic = list(self.prolog.query(f"explicatie({d}, E)"))
        return d, explic[0]["E"] if explic else "Fara explicatie"


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

        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y_class


class MLDatasetBuilder_Sequence:

    def __init__(self, window=30):
        self.window = window
        self.scaler = StandardScaler()

    def construieste_dataset(self, df):
        df = df.sort_values("Data").copy()
        features = ["Open", "High", "Low", "Close", "Volume"]
        data = df[features].values
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
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def genereaza_cu_debug(self, text_input, num_generate=5, temperature=0.7):
        print("\nInput text")
        print(text_input)

        # Tokenizare
        tokens = self.tokenizer.tokenize(text_input)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        print("\nTokenizare")
        print(f"Tokeni: {tokens[:5]}{'...' if len(tokens) > 5 else ''}")
        print(f"Număr tokeni: {len(tokens)}")

        # Pregătește input cu torch.no_grad()
        with torch.no_grad():
            input_ids = torch.tensor([token_ids])

            # Obține embeddings
            word_embeddings = self.model.transformer.wte(input_ids)

            print("\nWord Embeddings (primele 5 tokeni)")
            for i, token in enumerate(tokens[:5]):
                print(f"{token:>10} → shape: {word_embeddings[0, i].shape}")

            # Afișare detaliată primul embedding
            print("\nDetalii pentru primul token:")

            first_embedding = word_embeddings[0, 0]
            print(f"Token: '{tokens[0]}' (ID: {input_ids[0, 0].item()})")
            print(f"Shape: {first_embedding.shape}")
            print(f"\nPrimele 20 valori din vector (din 768 total):")
            for j in range(20):
                print(f"  Dim {j:3d}: {first_embedding[j]:.6f}")

            print(f"\nStatistici embedding:")
            print(f"  Min:    {first_embedding.min():.6f}")
            print(f"  Max:    {first_embedding.max():.6f}")
            print(f"  Mean:   {first_embedding.mean():.6f}")
            print(f"  Std:    {first_embedding.std():.6f}")
            print(f"  Norma L2: {torch.norm(first_embedding):.6f}")

            # Positional embeddings
            positions = torch.arange(len(token_ids)).unsqueeze(0)
            pos_embeddings = self.model.transformer.wpe(positions)
            hidden_states = word_embeddings + pos_embeddings

            # Transformer layers
            print("\nTransformer Layers")
            for idx, layer in enumerate(self.model.transformer.h[:3]):  # Doar primele 3 straturi
                hidden_states = layer(hidden_states)[0]
                print(f"Layer {idx + 1:02d} | Shape: {hidden_states.shape}")

            # LM Head
            logits = self.model.lm_head(hidden_states)
            next_token_logits = logits[0, -1, :]

            # Probabilități
            probs = torch.softmax(next_token_logits, dim=-1).detach().numpy()
            top_indices = np.argsort(probs)[-5:][::-1]

            print("\nTop 5 probabilități pentru următorul token")
            for idx in top_indices:
                token = self.tokenizer.decode([idx])
                prob = probs[idx]
                print(f"  [{repr(token)}] → {prob:.4f}")

        # Generare simplificată
        print("\nGenerare text")
        generated = self.genereaza(text_input, max_length=50)
        print(f"{generated[:100]}{'...' if len(generated) > 100 else ''}")

    def genereaza_analiza_nlp(self, simbol, metrici):
        context = (
            f"Analysis for {simbol}: "
            f"The stock price change is {metrici['evolutie_procent']:.2f}%. "
            f"The current trend is {metrici['trend']} and volatility is {metrici['volatilitate']:.2f}. "
            f"In my opinion, this means that"
        )

        insight = self.genereaza(context, max_length=100)

        return insight.replace(context, "").strip()

        return generated


# GROK AI
class GrokModel:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GROK_API_KEY")

    def genereaza(self, context):
        """Generează răspuns folosind Grok AI"""
        if not self.api_key:
            return "GROK API KEY lipsă – folosesc modul simulat."

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
                        "content": "Ești un asistent financiar profesionist. Răspunzi STRICT pe baza datelor furnizate."
                    },
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 600
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
            rezultate["Prolog"] = mapping[decizie_p]
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
        # Conversie la numpy
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Mapare clase la indexi pozitivi pentru sklearn
        # -1 -> 0, 0 -> 1, 1 -> 2
        y_true_mapped = y_true + 1
        y_pred_mapped = y_pred + 1

        metrici = {}

        # METRICI DE BAZĂ
        metrici['accuracy'] = accuracy_score(y_true, y_pred)
        metrici['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

        # METRICI PER CLASĂ (macro average)
        metrici['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrici['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrici['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # METRICI WEIGHTED (pentru clase dezechilibrate)
        metrici['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrici['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrici['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # METRICI SPECIFICE
        metrici['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        metrici['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

        # CONFUSION MATRIX
        metrici['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])

        # METRICI FINANCIARE SPECIFICE
        # Directional Accuracy (câte mișcări de preț au fost prezise corect)
        directional_acc = np.mean(np.sign(y_true) == np.sign(y_pred))
        metrici['directional_accuracy'] = directional_acc

        # Profit teoretich (simplificat: +1 pentru predicție corectă, -1 pentru greșită)
        profit_signals = (y_true == y_pred).astype(int) * 2 - 1
        metrici['cumulative_profit'] = np.sum(profit_signals)

        return metrici

    @staticmethod
    def afiseaza_raport_detaliat(metrici, model_name):
        print(f"\n{'=' * 60}")
        print(f"RAPORT PERFORMANȚĂ: {model_name}")
        print(f"{'=' * 60}")

        print(f"\nMETRICI GENERALE:")
        print(f"  • Accuracy:                {metrici['accuracy']:.4f}")
        print(f"  • Balanced Accuracy:       {metrici['balanced_accuracy']:.4f}")
        print(f"  • Directional Accuracy:    {metrici['directional_accuracy']:.4f}")

        print(f"\nMETRICI MACRO (per clasă):")
        print(f"  • Precision (macro):       {metrici['precision_macro']:.4f}")
        print(f"  • Recall (macro):          {metrici['recall_macro']:.4f}")
        print(f"  • F1-Score (macro):        {metrici['f1_macro']:.4f}")

        print(f"\nMETRICI WEIGHTED:")
        print(f"  • Precision (weighted):    {metrici['precision_weighted']:.4f}")
        print(f"  • Recall (weighted):       {metrici['recall_weighted']:.4f}")
        print(f"  • F1-Score (weighted):     {metrici['f1_weighted']:.4f}")

        print(f"\nMETRICI AVANSATE:")
        print(f"  • Matthews Correlation:    {metrici['matthews_corrcoef']:.4f}")
        print(f"  • Cohen's Kappa:           {metrici['cohen_kappa']:.4f}")

        print(f"\nMETRICI FINANCIARE:")
        print(f"  • Cumulative Profit:       {metrici['cumulative_profit']}")

        print(f"\nCONFUSION MATRIX:")
        cm = metrici['confusion_matrix']
        labels = ['VINDE(-1)', 'PĂSTREAZĂ(0)', 'CUMPĂRĂ(1)']
        print(f"              Predicted")
        print(f"              {labels[0]:>12} {labels[1]:>12} {labels[2]:>12}")
        for i, label in enumerate(labels):
            print(f"  Actual {label:>12}  {cm[i, 0]:>12} {cm[i, 1]:>12} {cm[i, 2]:>12}")


# TESTE STATISTICE COMPARATIVE
class StatisticalTests:

    @staticmethod
    def mcnemar_test(y_true, y_pred1, y_pred2, model1_name, model2_name):

        y_true = np.array(y_true)
        y_pred1 = np.array(y_pred1)
        y_pred2 = np.array(y_pred2)

        correct1 = (y_true == y_pred1)
        correct2 = (y_true == y_pred2)

        a = np.sum(correct1 & correct2)
        b = np.sum(correct1 & ~correct2)
        c = np.sum(~correct1 & correct2)
        d = np.sum(~correct1 & ~correct2)

        table = np.array([[a, b], [c, d]])

        # Aplicăm testul McNemar
        result = mcnemar(table, exact=True)

        print(f"\nMcNemar Test: {model1_name} vs {model2_name}")
        print(f"  Tabel contingență:")
        print(f"    Ambele corecte:     {a}")
        print(f"    Doar {model1_name} corect:  {b}")
        print(f"    Doar {model2_name} corect:  {c}")
        print(f"    Ambele greșite:     {d}")
        print(f"  Statistic: {result.statistic:.4f}")
        print(f"  P-value: {result.pvalue:.4f}")

        if result.pvalue < 0.05:
            print(f"  ✓ Există diferență SEMNIFICATIVĂ (p < 0.05)")
        else:
            print(f"  ✗ NU există diferență semnificativă (p ≥ 0.05)")

        return result

    @staticmethod
    def friedman_test(results_dict, y_true):

        # Construim matricea: fiecare rând = o instanță, fiecare coloană = un model
        model_names = list(results_dict.keys())
        n_samples = len(y_true)

        # Calculăm accuracy per sample pentru fiecare model
        accuracy_matrix = []
        for model_name in model_names:
            y_pred = results_dict[model_name]
            correct = (np.array(y_true) == np.array(y_pred)).astype(int)
            accuracy_matrix.append(correct)

        accuracy_matrix = np.array(accuracy_matrix).T  # transpose pentru format corect

        # Aplicăm testul Friedman
        statistic, pvalue = friedmanchisquare(*[accuracy_matrix[:, i] for i in range(len(model_names))])

        print(f"\nFriedman Test (comparație multiplă)")
        print(f"  Modele comparate: {len(model_names)}")
        print(f"  Sample size: {n_samples}")
        print(f"  Chi-square statistic: {statistic:.4f}")
        print(f"  P-value: {pvalue:.4f}")

        if pvalue < 0.05:
            print(f"  ✓ Există diferențe SEMNIFICATIVE între modele (p < 0.05)")
        else:
            print(f"  ✗ NU există diferențe semnificative între modele (p ≥ 0.05)")

        return statistic, pvalue

    @staticmethod
    def wilcoxon_signed_rank_test(y_true, y_pred1, y_pred2, model1_name, model2_name):

        # Test Wilcoxon pentru compararea pereche a două modele

        correct1 = (np.array(y_true) == np.array(y_pred1)).astype(int)
        correct2 = (np.array(y_true) == np.array(y_pred2)).astype(int)

        statistic, pvalue = wilcoxon(correct1, correct2, zero_method='wilcox')

        print(f"\nWilcoxon Signed-Rank Test: {model1_name} vs {model2_name}")
        print(f"  Statistic: {statistic:.4f}")
        print(f"  P-value: {pvalue:.4f}")

        if pvalue < 0.05:
            print(f"  ✓ Există diferență SEMNIFICATIVĂ (p < 0.05)")
        else:
            print(f"  ✗ NU există diferență semnificativă (p ≥ 0.05)")

        return statistic, pvalue


# VIZUALIZĂRI GRAFICE
class PerformanceVisualizer:

    @staticmethod
    def plot_comparative_metrics(metrics_dict, save_path=None):

        # Grafic comparativ cu principalele metrici pentru toate modelele

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Comparație Metrici de Performanță - Toate Modelele', fontsize=16, fontweight='bold')

        model_names = list(metrics_dict.keys())

        # Metrici de vizualizat
        metrics_to_plot = [
            ('accuracy', 'Accuracy'),
            ('balanced_accuracy', 'Balanced Accuracy'),
            ('f1_macro', 'F1-Score (Macro)'),
            ('precision_macro', 'Precision (Macro)'),
            ('recall_macro', 'Recall (Macro)'),
            ('matthews_corrcoef', 'Matthews Correlation')
        ]

        for idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]

            values = [metrics_dict[model][metric_key] for model in model_names]
            colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))

            bars = ax.bar(range(len(model_names)), values, color=colors, alpha=0.8, edgecolor='black')
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.set_ylabel('Score')
            ax.set_title(metric_label, fontweight='bold')
            ax.set_ylim([0, 1.0])
            ax.grid(axis='y', alpha=0.3)

            # Adăugăm valori pe bare
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_confusion_matrices(metrics_dict, save_path=None):

        # Grid cu confusion matrices pentru toate modelele

        n_models = len(metrics_dict)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        fig.suptitle('Confusion Matrices - Toate Modelele', fontsize=16, fontweight='bold')

        axes = axes.flatten() if n_models > 1 else [axes]

        labels = ['VINDE\n(-1)', 'PĂSTREAZĂ\n(0)', 'CUMPĂRĂ\n(1)']

        for idx, (model_name, metrics) in enumerate(metrics_dict.items()):
            cm = metrics['confusion_matrix']
            ax = axes[idx]

            # Normalizare pentru culori mai bune
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=labels, yticklabels=labels,
                        cbar_kws={'label': 'Număr predicții'})

            ax.set_title(f'{model_name}\nAccuracy: {metrics["accuracy"]:.3f}', fontweight='bold')
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')

        # Ascundem axele nefolosite
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_model_ranking(metrics_dict, save_path=None):

        model_names = list(metrics_dict.keys())

        # Calculăm score-uri agregat
        metrics_keys = ['accuracy', 'f1_macro', 'matthews_corrcoef', 'balanced_accuracy']

        scores = []
        for model in model_names:
            score = np.mean([metrics_dict[model][key] for key in metrics_keys])
            scores.append(score)

        # Sortăm descrescător
        sorted_indices = np.argsort(scores)[::-1]
        sorted_models = [model_names[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]

        fig, ax = plt.subplots(figsize=(12, 8))

        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_models)))
        bars = ax.barh(range(len(sorted_models)), sorted_scores, color=colors,
                       edgecolor='black', linewidth=1.5)

        ax.set_yticks(range(len(sorted_models)))
        ax.set_yticklabels(sorted_models)
        ax.set_xlabel('Score Agregat (avg: Accuracy, F1, MCC, Balanced Acc)', fontweight='bold')
        ax.set_title('Ranking Modele - Performanță Globală', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1.0])
        ax.grid(axis='x', alpha=0.3)

        # Adăugăm rank și score pe bare
        for idx, (bar, score) in enumerate(zip(bars, sorted_scores)):
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                    f'#{idx + 1} | {score:.4f}',
                    ha='left', va='center', fontweight='bold', fontsize=11)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_financial_metrics(metrics_dict, save_path=None):

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Metrici Financiare - Aplicabilitate în Trading', fontsize=14, fontweight='bold')

        model_names = list(metrics_dict.keys())

        # Directional Accuracy
        ax1 = axes[0]
        dir_acc = [metrics_dict[model]['directional_accuracy'] for model in model_names]
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(model_names)))

        bars1 = ax1.bar(range(len(model_names)), dir_acc, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.set_ylabel('Directional Accuracy')
        ax1.set_title('Acuratețea Direcției Mișcării Prețului', fontweight='bold')
        ax1.set_ylim([0, 1.0])
        ax1.axhline(y=0.5, color='red', linestyle='--', label='Random (50%)')
        ax1.grid(axis='y', alpha=0.3)
        ax1.legend()

        for bar, value in zip(bars1, dir_acc):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                     f'{value:.3f}', ha='center', va='bottom', fontsize=9)

        # Cumulative Profit (simplificat)
        ax2 = axes[1]
        cum_profit = [metrics_dict[model]['cumulative_profit'] for model in model_names]
        colors = ['green' if p > 0 else 'red' for p in cum_profit]

        bars2 = ax2.bar(range(len(model_names)), cum_profit, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xticks(range(len(model_names)))
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.set_ylabel('Profit Cumulat (simplificat)')
        ax2.set_title('Profit Teoretic Bazat pe Semnale Corecte', fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.grid(axis='y', alpha=0.3)

        for bar, value in zip(bars2, cum_profit):
            height = bar.get_height()
            va = 'bottom' if height > 0 else 'top'
            offset = 2 if height > 0 else -2
            ax2.text(bar.get_x() + bar.get_width() / 2., height + offset,
                     f'{int(value)}', ha='center', va=va, fontsize=9, fontweight='bold')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# SISTEM COMPLET DE EVALUARE
class ModelEvaluationOrchestrator:

    def __init__(self):
        self.metrics_calculator = PerformanceMetrics()
        self.statistical_tests = StatisticalTests()
        self.visualizer = PerformanceVisualizer()

    def evalueaza_complet(self, y_true, predictions_dict, output_dir="evaluation_results"):

        import os
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("SISTEM COMPLET DE EVALUARE PERFORMANȚĂ MODELE ML/DL")
        print("=" * 80)

        # CALCUL METRICI PENTRU FIECARE MODEL
        print("\n[1/4] Calculare metrici de performanță...")
        all_metrics = {}

        for model_name, y_pred in predictions_dict.items():
            metrics = self.metrics_calculator.calculeaza_metrici_complete(
                y_true, y_pred, model_name
            )
            all_metrics[model_name] = metrics
            self.metrics_calculator.afiseaza_raport_detaliat(metrics, model_name)

        # TESTE STATISTICE
        print("\n" + "=" * 80)
        print("[2/4] Teste statistice comparative...")
        print("=" * 80)

        # Friedman test (comparație multiplă)
        self.statistical_tests.friedman_test(predictions_dict, y_true)

        # McNemar pairwise (primele 3 modele pentru exemplu)
        model_list = list(predictions_dict.keys())
        if len(model_list) >= 2:
            self.statistical_tests.mcnemar_test(
                y_true,
                predictions_dict[model_list[0]],
                predictions_dict[model_list[1]],
                model_list[0],
                model_list[1]
            )

        # VIZUALIZĂRI
        print("\n" + "=" * 80)
        print("[3/4] Generare vizualizări...")
        print("=" * 80)

        self.visualizer.plot_comparative_metrics(
            all_metrics,
            save_path=f"{output_dir}/comparative_metrics.png"
        )

        self.visualizer.plot_confusion_matrices(
            all_metrics,
            save_path=f"{output_dir}/confusion_matrices.png"
        )

        self.visualizer.plot_model_ranking(
            all_metrics,
            save_path=f"{output_dir}/model_ranking.png"
        )

        self.visualizer.plot_financial_metrics(
            all_metrics,
            save_path=f"{output_dir}/financial_metrics.png"
        )

        # RAPORT FINAL
        print("\n" + "=" * 80)
        print("[4/4] Raport Final - Recomandări")
        print("=" * 80)

        self._genereaza_raport_final(all_metrics)

        print(f"\n✓ Evaluare completă! Grafice salvate în: {output_dir}/")

        return all_metrics

    def _genereaza_raport_final(self, metrics_dict):

        # Generează recomandări finale bazate pe metrici
        # Calculăm scoruri composite
        model_scores = {}
        for model, metrics in metrics_dict.items():
            score = (
                    metrics['accuracy'] * 0.25 +
                    metrics['f1_macro'] * 0.25 +
                    metrics['matthews_corrcoef'] * 0.25 +
                    metrics['directional_accuracy'] * 0.25
            )
            model_scores[model] = score

        # Sortare
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

        print("\nTOP 3 MODELE RECOMANDATE:")
        for idx, (model, score) in enumerate(sorted_models[:3], 1):
            print(f"\n  {idx}. {model}")
            print(f"     Score compozit: {score:.4f}")
            print(f"     Accuracy: {metrics_dict[model]['accuracy']:.4f}")
            print(f"     F1-Macro: {metrics_dict[model]['f1_macro']:.4f}")
            print(f"     Directional Acc: {metrics_dict[model]['directional_accuracy']:.4f}")

        print("\nRECOMANDĂRI:")
        best_model = sorted_models[0][0]
        print(f"  • Modelul {best_model} prezintă cea mai bună performanță globală")

        # Verificăm dacă există modele apropiate ca performanță
        if len(sorted_models) > 1:
            diff = sorted_models[0][1] - sorted_models[1][1]
            if diff < 0.02:
                print(f"  • {sorted_models[1][0]} este foarte aproape ca performanță - considerați ensemble")

        # Recomandare bazată pe directional accuracy
        best_dir_acc = max(metrics_dict.items(), key=lambda x: x[1]['directional_accuracy'])
        if best_dir_acc[0] != best_model:
            print(
                f"  • Pentru trading real, considerați {best_dir_acc[0]} (directional accuracy: {best_dir_acc[1]['directional_accuracy']:.4f})")


#  CONSENS
def consensus(rezultate, pret_curent):
    """
    Transformă toate predicțiile valide în clase [-1, 0, 1]
    și aplică vot majoritar.
    """
    rezultate_clase = {}

    for k, v in rezultate.items():
        # Ignorăm explicațiile
        if k == "Prolog_explicatie":
            continue

        # Orice model care nu știe → NU votează
        elif v is None:
            continue

        # Modele ML clasice (-1,0,1)
        else:
            rezultate_clase[k] = int(v)

    # Dacă nimeni nu a votat
    if not rezultate_clase:
        return None, rezultate_clase

    votes = list(rezultate_clase.values())
    final = max(set(votes), key=votes.count)

    return final, rezultate_clase


# AFISARE HEADER ȘI ISTORIC EVOLUȚIE MODELE AI
def afiseaza_istoric_AI():
    print("=" * 44)
    print("SISTEM INTEGRAT DE ANALIZĂ FINANCIARĂ AI")
    print("De la NaiveBayes (1763) la Grok AI (2024)")
    print("=" * 44)
    print("\nIstoric evoluție modele utilizate în ML/AI:\n")

    istoric = [
        (1763, "NaiveBayes", "Statistic / ML"),
        (1906, "MarkovChain", "Statistic / AI secvențial"),
        (1972, "Prolog", "AI simbolic / logic"),
        (1986, "NeuralNetwork", "Deep Learning / MLP"),
        (1997, "LSTM", "Deep Learning / AI secvențial"),
        (2001, "RandomForest", "ML clasic"),
        (2014, "GRU", "Deep Learning / AI secvențial"),
        (2016, "TCN", "Deep Learning / AI secvențial"),
        (2017, "Transformer", "Deep Learning modern / AI"),
    ]

    for an, descriere, tip in istoric:
        print(f"{an} ─ {descriere} [{tip}]")


# MAIN PIPELINE – NEXT-DAY PREDICTION PE COMPANII
if __name__ == "__main__":
    grok_api_key = os.getenv("GROK_API_KEY")
    afiseaza_istoric_AI()
    hf = HuggingFaceNER(COMPANII_AUTOMOTIVE)
    # Listează companiile disponibile
    print("\nCompanii disponibile în sistem:")
    for nume, simbol in hf.simboluri.items():
        print(f"  - {nume.title()}: {simbol}")
    db = BazaDate()
    ner = HuggingFaceNER(COMPANII_AUTOMOTIVE)
    yf_client = YahooFinance()
    data_manager = DataManager(db, ner, yf_client)
    data_manager.verifica_si_descarca_date(prag_min_randuri=200)

    exporter = ExportDate(db)
    predictor = NextDayPredictor(window=30, n_lags=5)
    analizator_numpy = AnalizaNumPy()

    intrebari = [
        "Care este evaluarea financiară a firmei TESLA azi?",
        # "Cum performează Ford pe piață?",
        # "Ce părere ai despre BMW?",
        # "Cum se descurcă Ferrari?",
    ]

    ROLLING_WINDOW = 900
    label_mapping = {-1: "VINDE", 0: "PASTREAZA", 1: "CUMPARA"}

    for intrebare in intrebari:
        print(f"\nÎntrebare: {intrebare}")

        # Identificare Companie (NER)
        nume_companie, simbol = ner.extrage_companie(intrebare)

        # Export Date și Info Fișier
        df, filehash = exporter.exporta(simbol, zile=ROLLING_WINDOW)
        if df is None or df.empty:
            print(f"Nu există date pentru simbolul {simbol}.")
            continue

        print(f"\n  Date disponibile: {len(df)} zile ({df['Data'].iloc[0]} → {df['Data'].iloc[-1]})")
        print(f"  CSV exportat: date_per_simbol/{simbol}_{ROLLING_WINDOW}_zile.csv")

        # HUGGING FACE (NER)
        print(f"  HUGGING FACE (NER) → Companie: {nume_companie}, Simbol: {simbol}")

        # YAHOO FINANCE (Date brute)
        pret_azi = float(df["Close"].iloc[-1])
        pret_ieri = float(df["Close"].iloc[-2])
        print(f"  YAHOO FINANCE → Preț ieri: ${pret_ieri:.2f}, Preț azi: ${pret_azi:.2f}")

        # NUMPY ANALYSIS (Calcul metrici statistice)
        metrici = analizator_numpy.calculeaza_metrici({
            "pret_ieri": pret_ieri,
            "pret_azi": pret_azi,
            "istoric_complet": df
        })

        print(f"  NUMPY ANALYSIS → Evoluție: {metrici['evolutie_procent']:+.2f}%, "
              f"Volatilitate: {metrici['volatilitate']:.2f}%, "
              f"Momentum: {metrici['momentum']:.2f}, "
              f"Trend: {metrici['trend']}")

        rezultate = predictor.ruleaza(df)
        pret_curent = df["Close"].iloc[-1]

        print("\nPredicții pentru ziua următoare (pe baza istoricului):")
        for model, pred in rezultate.items():
            if model == "Prolog_explicatie":
                continue

            if pred is None:
                print(f"  {model}: NECUNOSCUT")
            else:
                print(f"  {model}: {label_mapping[pred]}")

        decizie_finala, rezultate_clase = consensus(rezultate, pret_curent)

        if decizie_finala is None:
            print("\n⚠Nu s-a putut lua o decizie (prea puține predicții).")
        else:
            # Extragem toate voturile (doar valorile -1, 0, 1)
            toate_voturile = list(rezultate_clase.values())

            # Numărăm aparițiile pentru fiecare etichetă
            nr_cumpara = toate_voturile.count(1)
            nr_vinde = toate_voturile.count(-1)
            nr_pastreaza = toate_voturile.count(0)

            # Afișăm numărătoarea cerută
            print(f"\nCUMPĂRĂ = {nr_cumpara}, VINDE = {nr_vinde}, PĂSTREAZĂ = {nr_pastreaza}.")

            # Afișăm decizia finală
            print(f"Decizie finală (vot majoritar între modele): {label_mapping[decizie_finala]}")

        model_gpt2 = GPT2Model()
        model_grok = GrokModel(api_key=grok_api_key)

        print("\nGPT-2 PREDICTION WORKFLOW")
        context_gpt2 = f"Stock {simbol} moved {metrici['evolutie_procent']:.2f}% today."
        gpt2_debug = model_gpt2.genereaza_cu_debug(context_gpt2, num_generate=3, temperature=0.7)

        gpt2_text = model_gpt2.genereaza_analiza_nlp(simbol, metrici)

        gpt2_linii = textwrap.wrap(gpt2_text, width=100)
        for linie in gpt2_linii[:3]:
            print(f"  {linie}")
        if len(gpt2_linii) > 3:
            print(f"  [...] (text continuă)")

        print(f"\n • Grok AI Analysis:")
        # Pregătim contextul pentru Grok
        context_grok = f"Analizează simbolul {simbol}. Evoluție: {metrici['evolutie_procent']}%, Trend: {metrici['trend']}. Decizie consensus: {label_mapping[decizie_finala]}."
        grok_text = model_grok.genereaza(context_grok)

        grok_linii = textwrap.wrap(grok_text, width=100)
        for linie in grok_linii[:5]:
            print(f"  {linie}")
        if len(grok_linii) > 5:
            print(f"  [...] (text continuă)")

    # EVALUARE PERFORMANȚĂ
    print("\n" + "=" * 50)
    print("DEMARARE EVALUARE AUTOMATĂ MODELE")
    print("=" * 50)

    # Pregătim datele pentru backtesting
    X_eval, y_true_eval = predictor.tab_builder.construieste_dataset(df)

    # Folosim ultimele 50 de zile pentru testare
    X_test = X_eval[-50:]
    y_true = y_true_eval[-50:]
    X_train_eval = X_eval[:-50]
    y_train_eval = y_true_eval[:-50]

    # Re-antrenăm și colectăm predicții folosind clasele definite în script
    # Instanțiem local pentru a evita dependența de importuri externe
    model_rf = RFModel()
    model_rf.train(X_train_eval, y_train_eval)

    model_nb = NaiveBayesTabular()
    model_nb.train(X_train_eval, y_train_eval)

    model_nn = NNModel(input_dim=X_eval.shape[1])
    model_nn.train(X_train_eval, y_train_eval)

    dict_predictii = {
        "RandomForest": model_rf.predict(X_test),
        "NaiveBayes": model_nb.predict(X_test),
        "NeuralNetwork": model_nn.predict(X_test)
    }

    # Inițializăm Orchestratorul de Evaluare
    orchestrator = ModelEvaluationOrchestrator()

    # Rulăm evaluarea
    metrici_finale = orchestrator.evalueaza_complet(
        y_true=y_true,
        predictions_dict=dict_predictii,
        output_dir=f"evaluare_{simbol}"
    )
