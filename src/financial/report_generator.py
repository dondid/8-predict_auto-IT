
from fpdf import FPDF
import pandas as pd
import tempfile
import os

class FinancialReportPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Raport Analiza Financiara Auto Predictor', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 10, body)
        self.ln()

class ReportGenerator:
    @staticmethod
    def generate_pdf(data, metrics, predictions, consensus, ai_analysis, chart_image_path=None):
        pdf = FinancialReportPDF()
        pdf.add_page()
        
        # 1. Info Generale
        pdf.chapter_title(f"Analiza pentru: {data['company_name']} ({data['symbol']})")
        pdf.chapter_body(f"Pret Curent: ${metrics.get('pret_azi', 0):.2f}\n"
                         f"Evolutie: {metrics.get('evolutie_procent', 0):.2f}%\n"
                         f"Trend: {metrics.get('trend', 'N/A').upper()}\n"
                         f"Volatilitate: {metrics.get('volatilitate', 0):.2f}%")
                         
        # 2. Consensus
        pdf.chapter_title("Decizie Consensus")
        pdf.set_font('Arial', 'B', 14)
        if consensus == "CUMPARA":
            pdf.set_text_color(0, 128, 0)
        elif consensus == "VINDE":
            pdf.set_text_color(255, 0, 0)
        else:
            pdf.set_text_color(200, 200, 0)
        pdf.cell(0, 10, consensus, 0, 1, 'L')
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)

        # 3. Chart (if available)
        if chart_image_path and os.path.exists(chart_image_path):
            pdf.chapter_title("Evolutie Grafica")
            # Resize image to fit width (A4 width ~210mm, margins ~20mm => ~190mm)
            pdf.image(chart_image_path, x=10, w=190)
            pdf.ln(5)

        # 4. AI Analysis
        pdf.chapter_title("Analiza Inteligenta Artificiala")
        
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 10, "Grok AI / GPT-2 Insight:", 0, 1)
        pdf.set_font('Arial', '', 10)
        # Handle potential None
        if ai_analysis:
             # Sanitize text
            safe_text = ai_analysis.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 6, safe_text)
        else:
            pdf.multi_cell(0, 6, "Nu a fost generata nicio analiza AI.")
        pdf.ln(5)
        
        # 5. Tabel Predictii
        pdf.chapter_title("Detaliu Predictii Modele ML")
        pdf.set_font('Arial', 'B', 10)
        
        # Table Header
        pdf.cell(60, 8, "Model", 1)
        pdf.cell(60, 8, "Semnal", 1)
        pdf.ln()
        
        pdf.set_font('Arial', '', 10)
        for model, val in predictions.items():
            if model == "Prolog_explicatie": continue
            
            decision = "PASTREAZA"
            if val == 1: decision = "CUMPARA"
            elif val == -1: decision = "VINDE"
            
            pdf.cell(60, 8, str(model), 1)
            pdf.cell(60, 8, decision, 1)
            pdf.ln()

        # Output
        # Use temp directory
        tmp_dir = tempfile.gettempdir()
        file_path = os.path.join(tmp_dir, f"raport_{data['symbol']}.pdf")
        pdf.output(file_path)
        return file_path
