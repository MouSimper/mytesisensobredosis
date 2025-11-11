#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluación post-fine-tuning (Meta-Llama-3-8B-Instruct + QLoRA)
Métricas: BLEU, BERTScore, MoverScore, ROUGE-L. Guarda CSV con predicciones y métricas.
Ajusta las rutas base_model / finetuned_model_path / dataset_path si es necesario.
"""

import os
import csv
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
base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
finetuned_model_path = "results_qlora_llama3\checkpoint-80"   # ruta donde guardaste el adaptador QLoRA
dataset_path = "eval.jsonl"            # tu dataset jsonl (prompt, expected_answer)
device = 0 if torch.cuda.is_available() else -1

# -------------------------
# Cargar tokenizer, modelo base y adapter PEFT
# -------------------------
print("Cargando tokenizer y modelo base...")
tokenizer = AutoTokenizer.from_pretrained(base_model)
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
# Generar respuestas y recolectar referencias
# -------------------------
preds = []
refs = []
ids = []

print("Generando respuestas (esto puede tardar)...")
for i, ex in enumerate(ds):
    prompt = ex["prompt"]
    ref = ex["expected_answer"]
    ids.append(ex.get("id", i))

    # Construye input (usa formato instruct si prefieres)
    input_text = f"<s>[INST] {prompt} [/INST]"

    out = gen_pipe(input_text, max_new_tokens=128, do_sample=False, temperature=0.0)[0]["generated_text"]
    # Extraer la porción de respuesta si el formato incluye [/INST]
    if "[/INST]" in out:
        generated = out.split("[/INST]")[-1].strip()
    else:
        # si no tiene delimitador, intentar remover el prompt (fallback)
        generated = out.replace(input_text, "").strip()

    preds.append(generated)
    refs.append(ref)

print("Generación completada.")

# -------------------------
# Métricas: BLEU (sacrebleu), BERTScore, MoverScore, ROUGE-L
# -------------------------
print("Calculando métricas...")

# BLEU usando evaluate (sacrebleu por debajo)
bleu_metric = evaluate.load("bleu")
bleu_res = bleu_metric.compute(predictions=preds, references=[[r] for r in refs])
bleu_score = bleu_res.get("bleu", None)

# BERTScore (usa bert_score package para más control)
P, R, F1 = bert_score(preds, refs, lang="es", verbose=False)
bert_f1_mean = float(F1.mean().cpu().numpy())

# MoverScore (moverscore_v2)
idf_ref = get_idf_dict(refs)
idf_pred = get_idf_dict(preds)
mover_scores = word_mover_score(refs, preds, idf_ref, idf_pred)
mover_mean = float(np.mean(mover_scores))

# ROUGE-L (rouge_scorer)
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
out_csv = "eval_results_post_finetune.csv"
print(f"Guardando resultados en {out_csv} ...")
rows = []
for i, (id_, p, r) in enumerate(zip(ids, preds, refs)):
    # métricas por ejemplo: rougeL + mover (mover y bert a nivel ejemplo no devuelven proms directos aquí fácil)
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
# añade métricas globales al header del CSV
df.to_csv(out_csv, index=False, encoding="utf-8-sig")

# También guardar resumen global en un pequeño JSON
summary = {
    "n_examples": len(preds),
    "bleu": float(bleu_score),
    "moverscore": mover_mean,
    "bertscore_f1": bert_f1_mean,
    "rougeL_f1": rouge_l_mean
}
import json
with open("eval_summary_post_finetune.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Hecho. Archivos generados:")
print(" -", out_csv)
print(" - eval_summary_post_finetune.json")
