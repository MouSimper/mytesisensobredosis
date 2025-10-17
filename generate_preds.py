# generate_preds_descriptions.py
print("=== Script generate_preds_descriptions.py iniciado ===")

import os
import time
import pandas as pd
from chatbot_llama3_4bit_con_clasificador import llama_chat_generate

CSV_IN = "validacion_llama3.csv"            # Tu CSV de entrada con observaciones
CSV_OUT = "validacion_llama3_with_preds.csv"  # CSV de salida con predicciones

# Prompt ajustado para generar descripciones completas en inglés
PROMPT_TEMPLATE = (
    "You are an expert fire-door inspector. "
    "Given the observation, provide a short, clear, and professional sentence in English "
    "describing the defect.\n\n"
    "Observation: {obs}\n\nDescription:"
)

SLEEP_BETWEEN = 0.5  # segundos entre llamadas para no sobrecargar el modelo

def generate_predictions():
    df = pd.read_csv(CSV_IN, encoding="utf-8")

    # Asegurarse de que exista la columna para guardar predicciones
    if "prediccion_modelo" not in df.columns:
        df["prediccion_modelo"] = ""

    for i, row in df.iterrows():
        if str(row["prediccion_modelo"]).strip():
            print(f"[{i}] Skip (ya tiene predicción)")
            continue

        entrada = str(row["entrada"])
        prompt = PROMPT_TEMPLATE.format(obs=entrada)
        try:
            print(f"[{i}] Generando para: {entrada[:80]}...")
            out = llama_chat_generate(prompt, history_pairs=None)
            pred = out.strip()
            print(f"[{i}] => {pred[:200]}")
            df.at[i, "prediccion_modelo"] = pred

            # Guardado incremental cada 5 filas
            if i % 5 == 0:
                df.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")

        except Exception as e:
            print(f"[{i}] Error generando: {e}")
            df.at[i, "prediccion_modelo"] = ""

        time.sleep(SLEEP_BETWEEN)

    df.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    print("Finalizado. Salida guardada en:", CSV_OUT)

if __name__ == "__main__":
    generate_predictions()
