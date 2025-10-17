# evaluate_llm_with_expanded_refs.py
print("=== Script evaluate_llm_with_expanded_refs.py iniciado ===")
import pandas as pd
import numpy as np
from evaluate import load
from sentence_transformers import SentenceTransformer, util

CSV_IN = "validacion_llama3_with_expanded_refs.csv"

# ============================
# 1️⃣ CARGAR CSV
# ============================
df = pd.read_csv(CSV_IN)
# Asegurarse que tenemos las columnas necesarias
if "prediccion_modelo" not in df.columns or "referencia_expanded" not in df.columns:
    raise ValueError("El CSV debe contener 'prediccion_modelo' y 'referencia_expanded'.")

# Filtrar filas válidas
df_valid = df.dropna(subset=["prediccion_modelo", "referencia_expanded"])
preds = df_valid["prediccion_modelo"].tolist()
refs = df_valid["referencia_expanded"].tolist()

# ============================
# 2️⃣ BERTSCORE
# ============================
print("\nCalculando BERTScore...")
bertscore = load("bertscore")
bertscore_res = bertscore.compute(predictions=preds, references=refs, lang="en")
bert_f1 = float(np.mean(bertscore_res["f1"]))
print(f"BERTScore (F1 promedio): {bert_f1:.4f}")

# ============================
# 3️⃣ COSINE SIMILARITY
# ============================
print("\nCalculando similitud de coseno (Sentence Transformers)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
emb_ref = model.encode(refs, convert_to_tensor=True)
emb_pred = model.encode(preds, convert_to_tensor=True)
cosine_scores = util.cos_sim(emb_pred, emb_ref).diagonal().cpu().numpy()
cosine_mean = float(np.mean(cosine_scores))
print(f"Similitud de coseno promedio: {cosine_mean:.4f}")

# ============================
# 4️⃣ DIVERSIDAD (DISTINCT)
# ============================
print("\nCalculando métricas de diversidad (Distinct-1 y Distinct-2)...")
def distinct_n(sentences, n=1):
    ngrams_set = set()
    total_ngrams = 0
    for sent in sentences:
        tokens = sent.split()
        ngrams_sent = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        ngrams_set.update(ngrams_sent)
        total_ngrams += len(ngrams_sent)
    if total_ngrams == 0:
        return 0.0
    return len(ngrams_set) / total_ngrams

distinct1 = distinct_n(preds, n=1)
distinct2 = distinct_n(preds, n=2)
print(f"Distinct-1 (unigrama): {distinct1:.4f}")
print(f"Distinct-2 (bigrama): {distinct2:.4f}")

# ============================
# 5️⃣ LONGITUD MEDIA DE RESPUESTA
# ============================
lengths = [len(p.split()) for p in preds]
mean_len = float(np.mean(lengths))
std_len = float(np.std(lengths))
print(f"Longitud media de respuesta: {mean_len:.1f} ± {std_len:.1f} palabras")

# ============================
# 6️⃣ RESUMEN FINAL
# ============================
print("\n=== RESULTADOS DE EVALUACIÓN CON REFERENCIAS EXPANDIDAS ===")
print(f"BERTScore (F1):          {bert_f1:.4f}")
print(f"Cosine Similarity:       {cosine_mean:.4f}")
print(f"Distinct-1:              {distinct1:.4f}")
print(f"Distinct-2:              {distinct2:.4f}")
print(f"Longitud media respuesta:{mean_len:.1f} ± {std_len:.1f}")
print("\nEvaluación completada ✅")
