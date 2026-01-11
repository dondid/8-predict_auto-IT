import os
from pypdf import PdfReader
from pathlib import Path

pdf_dir = Path("Machine Learning")
output_file = "course_content_summary.txt"

with open(output_file, "w", encoding="utf-8") as out:
    for pdf_file in sorted(pdf_dir.glob("*.pdf")):
        try:
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            out.write(f"=== {pdf_file.name} ===\n")
            out.write(text[:2000]) # Read first 2000 chars of each to get the gist/requirements
            out.write("\n\n")
            print(f"Computed {pdf_file.name}")
        except Exception as e:
            print(f"Error reading {pdf_file.name}: {e}")
