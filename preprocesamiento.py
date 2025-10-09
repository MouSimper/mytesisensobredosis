import pandas as pd
import re

def clean_text(text, lowercase=True, remove_urls=True, remove_emojis=True):
    """
    Función de limpieza general:
      - Quita etiquetas HTML
      - Quita URLs
      - Quita emojis (opcional)
      - Quita caracteres especiales no deseados
      - Reduce espacios múltiples y strip()
      - Convierte a minúsculas (opcional)
    """
    if not isinstance(text, str):
        return ""

    # Quitar etiquetas HTML
    text = re.sub(r'<[^>]+>', ' ', text)

    # Quitar URLs
    if remove_urls:
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Quitar emojis
    if remove_emojis:
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            "]+", flags=re.UNICODE
        )
        text = emoji_pattern.sub(r'', text)

    # Quitar caracteres especiales no deseados
    text = re.sub(r"[^a-zA-Z0-9.,;!?'\s]", '', text)

    # Reducir espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()

    # Minúsculas
    if lowercase:
        text = text.lower()

    return text

# --------------- USO ---------------
# 1) Carga tu archivo
df = pd.read_excel('dataset_balanceado.xlsx')

# 2) Aplica limpieza a la columna de texto
df['English'] = df['English'].astype(str).apply(lambda x: clean_text(x))

# 3) Guarda limpio
df.to_excel('dataset_balanceado_limpio.xlsx', index=False)
print("✅ Archivo limpio guardado como 'dataset_balanceado_limpio.xlsx'")
