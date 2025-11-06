# generate_preds_with_lora.py
print("=== Script generate_preds_with_lora.py iniciado ===")

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time

# ---------------- CONFIG ----------------
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
LORA_ADAPTER = "./qlora_llama3_test_adapter/checkpoint-300"  # usa tu último checkpoint
CSV_IN = "validacion_llama3.csv"
CSV_OUT = "validacion_llama3_with_lora_preds.csv"
PROMPT_TEMPLATE = (
    "You are an expert inspector. Given the observation, produce a short, "
    "clear diagnostic sentence in English describing the defect.\n\n"
    "Observation: {obs}\n\nDiagnosis:"
)
SLEEP_BETWEEN = 0.5  # segundos entre llamadas
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.2
TOP_P = 0.9
# ----------------------------------------

print("Cargando modelo base y LoRA adapter...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
model = PeftModel.from_pretrained(model, LORA_ADAPTER)
model.eval()

df = pd.read_csv(CSV_IN, encoding="utf-8")
if "prediccion_modelo" not in df.columns:
    df["prediccion_modelo"] = ""

for i, row in df.iterrows():
    if str(row["prediccion_modelo"]).strip():
        print(f"[{i}] Skip (already has prediction)")
        continue

    entrada = str(row["entrada"])
    prompt = PROMPT_TEMPLATE.format(obs=entrada)
    print(f"[{i}] Generating for: {entrada[:80]}...")

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        pred = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        df.at[i, "prediccion_modelo"] = pred.strip()
        print(f"[{i}] => {pred[:200]}")
    except Exception as e:
        print(f"[{i}] Error generating: {e}")
        df.at[i, "prediccion_modelo"] = ""

    # guarda incrementalmente
    if i % 5 == 0:
        df.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")

    time.sleep(SLEEP_BETWEEN)

df.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
print("Finished. Output saved to:", CSV_OUT)
