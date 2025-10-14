# convert_excel_to_jsonl.py
import pandas as pd
df = pd.read_excel("balanced_t5.xlsx")

# EJEMPLO: cada muestra será un prompt con la observación y la respuesta será la etiqueta.
# Ajusta template si deseas que genere un informe más largo.
def make_pair(obs, label):
    prompt = f"Observation: {obs}\n\nClassify into a label (one short label)."
    completion = " " + label  # leading space helpful for tokenizer when training causal LM
    return {"prompt": prompt, "completion": completion}

out = []
for _, row in df.iterrows():
    obs = str(row["English_clean"]).strip()
    label = str(row["Classification_English"]).strip()
    if obs and label:
        out.append(make_pair(obs, label))

import json
with open("dataset_sft.jsonl", "w", encoding="utf-8") as fh:
    for item in out:
        json.dump(item, fh, ensure_ascii=False)
        fh.write("\n")

print("Saved dataset_sft.jsonl with", len(out), "examples")
