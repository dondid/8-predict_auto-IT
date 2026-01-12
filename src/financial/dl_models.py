
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import os
import requests

# 1986 ─ NeuralNetwork (Backpropagation)
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
