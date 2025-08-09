import pandas as pd
import json
from PyPDF2 import PdfReader
import base64
import subprocess
import os

def load_csv_or_excel(file_path: str) -> pd.DataFrame:
    """Loads a CSV or Excel file into a pandas DataFrame."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type for tabular data.")

def load_pdf_text(file_path: str) -> str:
    """Extracts text from a PDF file."""
    text = ""
    with open(file_path, 'rb') as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

def read_text_file(file_path: str) -> str:
    """Reads a generic text file (e.g., .py, .sql, .txt)."""
    with open(file_path, 'r') as f:
        return f.read()

def encode_image_to_base64(image_path: str) -> str:
    """Encodes an image file into a base64 string for multi-modal prompts."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def run_user_script_sandboxed(script_path: str) -> str:
    """
    ⚠️ DANGEROUS: Executes a user-provided Python script.
    This should ideally run in a sandboxed Docker container.
    This simplified version runs it locally with a timeout.
    """
    try:
        result = subprocess.run(
            ["python", script_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=60 # Safety timeout
        )
        return f"--- Script Output ---\n{result.stdout}"
    except subprocess.CalledProcessError as e:
        return f"--- Script Error ---\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
    except Exception as e:
        return f"--- Execution Error ---\n{str(e)}"
