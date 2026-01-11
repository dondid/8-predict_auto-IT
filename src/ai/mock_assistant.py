import pandas as pd
import numpy as np
import random

class MockAI:
    def __init__(self, df):
        self.df = df
        self.context = self._build_context()
        
    def _build_context(self):
        return {
            "count": len(self.df),
            "mean_price": self.df['price'].mean(),
            "max_price": self.df['price'].max(),
            "min_price": self.df['price'].min(),
            "brands": self.df['make'].unique().tolist(),
            "correlations": self.df.select_dtypes(include=[np.number]).corr()['price'].sort_values(ascending=False)
        }
        
    def generate_response(self, query):
        query = query.lower()
        
        # Keyword matching logic
        if "pret" in query or "price" in query or "cost" in query:
            if "mediu" in query or "average" in query:
                return f"Prețul mediu al mașinilor din setul de date este ${self.context['mean_price']:,.2f}."
            if "maxim" in query or "scump" in query or "mare" in query:
                expensive = self.df.loc[self.df['price'].idxmax()]
                return f"Cea mai scumpă mașină este un {expensive['make']} {expensive['body-style']} la prețul de ${expensive['price']:,.0f}."
            if "minim" in query or "ieftin" in query or "mic" in query:
                cheap = self.df.loc[self.df['price'].idxmin()]
                return f"Cea mai ieftină mașină este un {cheap['make']} {cheap['body-style']} la prețul de ${cheap['price']:,.0f}."
            return f"Prețurile variază între ${self.context['min_price']:,.0f} și ${self.context['max_price']:,.0f}. Prețul mediu este ${self.context['mean_price']:,.0f}."
            
        if "cate" in query or "how many" in query:
            return f"Setul de date conține un total de {self.context['count']} mașini."
            
        if "brand" in query or "marca" in query or "marci" in query:
             return f"Mărcile disponibile sunt: {', '.join(self.context['brands'][:5])} și altele (total {len(self.context['brands'])})."
             
        if "corelat" in query or "influenta" in query or "factor" in query:
            top_corr = self.context['correlations'].index[1] # 0 is price itself
            val = self.context['correlations'][1]
            return f"Factorul care influențează cel mai mult prețul este '{top_corr}' (corelație {val:.2f})."
            
        if "bmw" in query:
            avg_bmw = self.df[self.df['make'] == 'bmw']['price'].mean()
            return f"BMW este un brand premium în acest dataset. Prețul mediu pentru un BMW este ${avg_bmw:,.0f}."
            
        # Fallback
        return "⚠️ [Demo Mode] Întrebare nerecunoscută. Întreabă-mă despre 'preț', 'mărci', 'cea mai scumpă mașină' sau 'factori de influență'."
