# expand_references_real.py
print("=== Script expand_references_real.py iniciado ===")
import os
import time
import pandas as pd

# Ajusta el nombre del módulo si tu archivo se llama distinto.
# Debe exponer llama_chat_generate(user_message, history_pairs=None)
from chatbot_llama3_4bit_con_clasificador import llama_chat_generate

CSV_IN = "validacion_llama3_with_preds.csv"
CSV_OUT = "validacion_llama3_with_expanded_refs.csv"

# Prompt para generar descripciones completas
PROMPT_TEMPLATE = (
    "You are an expert inspector. Given the short defect label, produce a short, "
    "clear sentence in English that could appear in an official report.\n\n"
    "Defect label: {label}\n\nDescription:"
)

SLEEP_BETWEEN = 0.5  # segundos entre llamadas

def expand_references():
    df = pd.read_csv(CSV_IN, encoding="utf-8")
    if "referencia_expanded" not in df.columns:
        df["referencia_expanded"] = ""

    for i, row in df.iterrows():
        label = str(row["referencia_humana"])
        if str(row.get("referencia_expanded", "")).strip():
            print(f"[{i}] Skip (already expanded)")
            continue

        prompt = PROMPT_TEMPLATE.format(label=label)
        try:
            print(f"[{i}] Generating description for: {label}")
            out = llama_chat_generate(prompt, history_pairs=None)
            desc = out.strip()
            print(f"[{i}] => {desc[:200]}")
            df.at[i, "referencia_expanded"] = desc
            # Guardar incrementalmente cada N filas
            if i % 5 == 0:
                df.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
        except Exception as e:
            print(f"[{i}] Error generating: {e}")
            df.at[i, "referencia_expanded"] = ""
        time.sleep(SLEEP_BETWEEN)

    df.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    print("Finished. Output saved to:", CSV_OUT)

if __name__ == "__main__":
    expand_references()
