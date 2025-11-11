#!/usr/bin/env python3
"""
evaluar_llm_sin_perplexity.py
Evaluación automática de LLaMA-3-8B-Instruct usando BLEU, MoverScore, BERTScore y ROUGE-L.
Entrada: JSONL con campos `id` (opcional), `prompt`, `expected_answer`.
Salida: CSV con predicciones y métricas agregadas.
"""

import argparse, json, os
from tqdm import tqdm
import pandas as pd
import torch
import numpy as np
if not hasattr(np, 'float'):
    np.float = float  # Fix para compatibilidad con moverscore_v2 y numpy >=1.24

from transformers import AutoTokenizer, AutoModelForCausalLM
import sacrebleu
from bert_score import score as bert_score
from moverscore_v2 import get_idf_dict, word_mover_score
from rouge_score import rouge_scorer


# =============== CONFIGURACIÓN DEL MODELO ===============
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN")  # opcional

MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 400
TEMPERATURE = 0.3
TOP_P = 0.8
TOP_K = 30
REPETITION_PENALTY = 1.15
BATCH_SIZE = 16
# =========================================================


def load_jsonl(path):
    """Carga el dataset JSONL (una línea = un ejemplo)."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def evaluate(args):
    """Evalúa el modelo con el dataset dado."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, use_fast=True, use_auth_token=HF_TOKEN
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", use_auth_token=HF_TOKEN
    )
    model.eval()

    dataset = load_jsonl(args.dataset)
    print(f"[INFO] Cargados {len(dataset)} ejemplos desde {args.dataset}")

    preds, refs, ids = [], [], []
    rows = []

    for item in tqdm(dataset, desc="Generando respuestas"):
        prompt = item.get("prompt", "")
        expected = item.get("expected_answer", "")
        id_ = item.get("id", "")

        # === Construcción de prompt con instrucción ===
        prompt_with_instr = (
            f"Pregunta: {prompt}\n"
            "Instrucción: Responde en español, en tono técnico y profesional. "
            "Escribe máximo 2 frases completas. "
            "Sé directo y conciso, usando terminología técnica exacta relacionada con cierrapuertas, bisagras, pestillos, burletes o puertas cortafuego según corresponda. "
            "Emplea imperativos siempre que sea posible (por ejemplo: 'Ajuste...', 'Revise...', 'Aplique...')."
        )

        inp = tokenizer(
            prompt_with_instr,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS
        ).to(model.device)

        input_len = inp["input_ids"].shape[1]

        # === Generación de respuesta ===
        with torch.no_grad():
            out = model.generate(
                **inp,
                do_sample=True,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                repetition_penalty=REPETITION_PENALTY,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        # === Decodificación limpia (sin eco del prompt) ===
        gen_ids = out[0]
        new_tokens = gen_ids[input_len:]
        gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        preds.append(gen_text)
        refs.append(expected)
        ids.append(id_)

        rows.append({
            "id": id_,
            "prompt": prompt,
            "expected": expected,
            "prediction": gen_text,
        })

    # =============== Métricas globales ===============
    print("\n[INFO] Calculando métricas globales...")

    # ✅ BLEU (orden corregido)
    bleu = sacrebleu.corpus_bleu(refs, [preds]).score

    # MoverScore
    print("[INFO] Calculando MoverScore...")
    idf_ref = get_idf_dict(refs)
    idf_pred = get_idf_dict(preds)
    mover_scores = word_mover_score(refs, preds, idf_ref, idf_pred)
    avg_mover = float(sum(mover_scores) / len(mover_scores))

    # BERTScore
    print("[INFO] Calculando BERTScore...")
    P, R, F1 = bert_score(preds, refs, lang="es", verbose=True)
    bert_f1_mean = float(F1.mean().cpu().numpy())

    # ROUGE-L
    print("[INFO] Calculando ROUGE-L...")
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = [scorer.score(r, p)['rougeL'].fmeasure for r, p in zip(refs, preds)]
    rouge_l_mean = float(np.mean(rouge_l_scores))

    # Guardar resultados
    df = pd.DataFrame(rows)
    df["bleu_corpus"] = bleu
    df["moverscore"] = avg_mover
    df["bert_f1_mean"] = bert_f1_mean
    df["rougeL_f1_mean"] = rouge_l_mean
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    # Mostrar resumen
    print("\n========= RESUMEN GLOBAL =========")
    print(f"Ejemplos evaluados: {len(preds)}")
    print(f"BLEU (corpus): {bleu:.2f}")
    print(f"MoverScore (promedio): {avg_mover:.3f}")
    print(f"BERTScore F1 (promedio): {bert_f1_mean:.3f}")
    print(f"ROUGE-L (promedio F1): {rouge_l_mean:.3f}")
    print(f"Resultados guardados en: {args.out}")
    print("==================================")

    return df


# =============== CLI =======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Ruta del dataset JSONL (prompt, expected_answer).")
    parser.add_argument("--out", default="eval_resultados.csv", help="Archivo CSV de salida.")
    args = parser.parse_args()
    evaluate(args)
