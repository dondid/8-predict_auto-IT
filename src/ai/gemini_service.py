import google.generativeai as genai
from src.config import GEMINI_API_KEY
from src.utils import get_logger
import pandas as pd
from io import StringIO

logger = get_logger(__name__)

class GeminiService:
    def __init__(self):
        self.api_key = GEMINI_API_KEY
        self.is_configured = False
        self.model = None
        
        if self.api_key and len(self.api_key) > 10:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel("gemini-1.5-flash")
                self.is_configured = True
            except Exception as e:
                logger.error(f"Failed to configure Gemini: {e}")

    def get_chat_response(self, user_input, history, context_df):
        """
        Generates a chat response using history and dataframe context.
        """
        if not self.is_configured:
            # Fallback to MockAI if Gemini is not available
            from src.ai.mock_assistant import MockAI
            mock = MockAI(context_df)
            return mock.generate_response(user_input)

        # Build Context Prompt (once or dynamically)
        # We assume history maintains the context, but let's reinforce if needed.
        # Actually, if we pass history which includes the system prompt, it's fine.
        
        # We will use a chat session logic managed by the caller usually, 
        # but here we can just do a stateless generation or formatted chat.
        # For simplicity with Streamlit, we often regenerate the session.
        
        try:
            chat = self.model.start_chat(history=history)
            response = chat.send_message(user_input)
            return response.text
        except Exception as e:
            return f"Eroare AI: {e}"

    def generate_brand_report(self, brand, brand_stats, modern_stats=None):
        """
        Generates the detailed brand encyclopedia report.
        brand_stats: dict of stats (price, hp, etc.)
        modern_stats: dict of modern stats (avg_price_2024, etc.) - Optional
        """
        if not self.is_configured:
            return "⚠️ Raport indisponibil. Activează Online Mode."

        context_str = f"Date 1985: Preț Mediu ${brand_stats.get('price', 0):.0f}, HP Max {brand_stats.get('hp', 0)}."
        if modern_stats:
            context_str += f"\nDate 2024 (Context Modern): Preț Mediu ${modern_stats.get('price', 0):.0f}, Modele Recente: {modern_stats.get('models', 'N/A')}."
            if 'live_market' in modern_stats:
                context_str += f"\n🔴 LIVE MARKET DATA (Yahoo Finance): {modern_stats['live_market']}"

        prompt = f"""
        Analizează brandul auto {brand}.
        CONTEXT DATE:
        {context_str}
        
        Sarcina ta:
        Generază 2 lucruri distincte:
        
        PARTEA 1: O analiză text (Markdown) despre:
        - Istoric și Reputație (de la origini până în prezent).
        - Evoluția Prețurilor (compară 1985 cu 2024 dacă ai date).
        - Analiză LIVE Market (comentează evoluția acțiunilor și știrile recente dacă există în context).
        - Probleme Tehnice Istorice vs Moderne.
        - Verdict: Merită colecționat un model clasic? Dar unul nou?
        
        PARTEA 2: Un tabel CSV (fără header) cu estimarea valorii medii (1985-2025).
        Format: An,Valoare_Estimata_USD (ex: 1985,5000)
        Ani: 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025.
        separator: "---CSV_START---"
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Eroare Generare Raport: {e}"

    def parse_evolution_csv(self, report_text):
        """
        Extracts the CSV part from the report text.
        Returns: (text_part, dataframe)
        """
        if "---CSV_START---" in report_text:
            text_part, csv_part = report_text.split("---CSV_START---")
            try:
                df = pd.read_csv(StringIO(csv_part.strip()), names=["An", "Valoare"])
                return text_part, df
            except:
                return text_part, None
        return report_text, None
