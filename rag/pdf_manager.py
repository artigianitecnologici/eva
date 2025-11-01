# File: pdf_manager.py
# Descrizione: Gestione dei pdf

import os
import shutil

PDF_DIR = 'data/pdfs'
os.makedirs(PDF_DIR, exist_ok=True)


def save_pdf(file):
    os.makedirs(PDF_DIR, exist_ok=True)
    filepath = os.path.join(PDF_DIR, file.filename)
    file.save(filepath)
    return os.path.abspath(filepath)

def list_pdfs():
    return os.listdir(PDF_DIR)

def delete_pdf(filename):
    path = os.path.join(PDF_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False
