import os  # <-- Importado (ya estaba, pero es necesario para la Solución B)
import csv
import json
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import evaluate
from bert_score import score as bert_score
from moverscore_v2 import get_idf_dict, word_mover_score
from rouge_score import rouge_scorer

# -------------------------
# Configuración
# -------------------------
# AJUSTA ESTAS RUTAS
base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
finetuned_model_path = "results_qlora_llama3/checkpoint-80"
dataset_path = "eval.jsonl"
device = 0 if torch.cuda.is_available() else -1

# Define tu batch_size aquí
EVAL_BATCH_SIZE = 8

# -------------------------
# Cargar tokenizer, modelo base y adapter PEFT
# -------------------------
print("Cargando tokenizer y modelo base...")
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Configurar padding para batching
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

print("Cargando adaptador PEFT (QLoRA)...")
model = PeftModel.from_pretrained(base, finetuned_model_path, device_map="auto")
model.eval()

# -------------------------
# Pipeline de generación
# -------------------------
gen_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    return_full_text=True,
)

# -------------------------
# Cargar dataset
# -------------------------
ds = load_dataset("json", data_files=dataset_path, split="train")
n_eval = len(ds)
print(f"Ejemplos en dataset: {n_eval}")

# -------------------------
# Generar respuestas (Sección modificada para Batching)
# -------------------------

print("Preparando datos para batching...")
prompts_formateados = []
refs = []
ids = []

for i, ex in enumerate(ds):
    prompt = ex["prompt"]
    refs.append(ex["expected_answer"])
    ids.append(ex.get("id", i))

    # Usar apply_chat_template
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    prompts_formateados.append(input_text)

print(f"Generando {len(prompts_formateados)} respuestas (en lotes de {EVAL_BATCH_SIZE})...")

preds = []
with torch.no_grad():
    # Llama al pipeline UNA SOLA VEZ con todos los prompts
    outputs = gen_pipe(
        prompts_formateados,
        max_new_tokens=128,
        do_sample=False,
        temperature=0.0,
        batch_size=EVAL_BATCH_SIZE,
        eos_token_id=tokenizer.eos_token_id
    )

print("Procesando salidas generadas...")
# Procesar los resultados del lote
for i, (out, prompt_text) in enumerate(zip(outputs, prompts_formateados)):
    full_text = out[0]["generated_text"]

    if "[/INST]" in full_text:
        generated = full_text.split("[/INST]")[-1].strip()
    else:
        generated = full_text.replace(prompt_text, "").strip()

    generated = generated.replace(tokenizer.eos_token, "").strip()
    preds.append(generated)

print("Generación completada.")

# -------------------------
# Métricas: BLEU (sacrebleu), BERTScore, MoverScore, ROUGE-L
# -------------------------
print("Calculando métricas...")

# BLEU
bleu_metric = evaluate.load("bleu")
bleu_res = bleu_metric.compute(predictions=preds, references=[[r] for r in refs])
bleu_score = bleu_res.get("bleu", None)

# BERTScore
P, R, F1 = bert_score(preds, refs, lang="es", verbose=False)
bert_f1_mean = float(F1.mean().cpu().numpy())

# MoverScore
safe_refs = [r if r else " " for r in refs]
safe_preds = [p if p else " " for p in preds]
idf_ref = get_idf_dict(safe_refs)
idf_pred = get_idf_dict(safe_preds)
mover_scores = word_mover_score(safe_refs, safe_preds, idf_ref, idf_pred)
mover_mean = float(np.mean(mover_scores))

# ROUGE-L
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
rouge_l_list = [scorer.score(r, p)["rougeL"].fmeasure for r, p in zip(refs, preds)]
rouge_l_mean = float(np.mean(rouge_l_list))

# -------------------------
# Imprimir resumen
# -------------------------
print("\n========= RESUMEN GLOBAL =========")
print(f"Ejemplos evaluados: {len(preds)}")
print(f"BLEU: {bleu_score:.4f}")
print(f"MoverScore (promedio): {mover_mean:.4f}")
print(f"BERTScore F1 (promedio): {bert_f1_mean:.4f}")
print(f"ROUGE-L (promedio F1): {rouge_l_mean:.4f}")
print("==================================")

# -------------------------
# Guardar CSV con predicciones + métricas por ejemplo
# -------------------------

### CAMBIOS SOLUCIÓN B ###
# 1. Crear la carpeta "outputs" si no existe
os.makedirs("outputs", exist_ok=True)

# 2. Definir la ruta del CSV para que apunte a "outputs"
out_csv = "outputs/eval_results_post_finetune.csv"
print(f"Guardando resultados en {out_csv} ...")

rows = []
for i, (id_, p, r) in enumerate(zip(ids, preds, refs)):
    rouge_l = rouge_l_list[i]
    mover = mover_scores[i]
    rows.append({
        "id": id_,
        "prompt": ds[i]["prompt"],
        "reference": r,
        "prediction": p,
        "rougeL_f": rouge_l,
        "moverscore": mover
    })

df = pd.DataFrame(rows)
df.to_csv(out_csv, index=False, encoding="utf-8-sig")

# Guardar resumen global en un JSON
summary = {
    "n_examples": len(preds),
    "bleu": float(bleu_score) if bleu_score is not None else 0.0,
    "moverscore": mover_mean,
    "bertscore_f1": bert_f1_mean,
    "rougeL_f1": rouge_l_mean
}

### CAMBIOS SOLUCIÓN B ###
# 3. Definir la ruta del JSON para que apunte a "outputs"
out_json = "outputs/eval_summary_post_finetune.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Hecho. Archivos generados:")
# 4. Actualizar los mensajes de salida
print(f" - {out_csv}")
print(f" - {out_json}")

