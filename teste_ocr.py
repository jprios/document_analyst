from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import os

# Caminho para o executável do Tesseract (ajuste conforme necessário)
pytesseract.pytesseract.tesseract_cmd = fr"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Caminho para o Poppler (ajuste se estiver em outro local)
POPPLER_PATH = fr"C:\Program Files\poppler-24.08.0\Library\bin"  # substitua pelo seu caminho real

# Caminho para o PDF de teste
caminho_pdf = fr"C:\Users\joao.rios\chatbot_documentos\pdf_digitalizado.pdf"  # substitua pelo seu PDF

# Converter páginas do PDF em imagens
imagens = convert_from_path(caminho_pdf, poppler_path=POPPLER_PATH)

# Fazer OCR em cada imagem (página)
for i, imagem in enumerate(imagens):
    texto = pytesseract.image_to_string(imagem, lang='por')
    print(f"\n========== TEXTO DA PÁGINA {i+1} ==========\n")
    print(texto)
