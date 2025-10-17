#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de evaluación con referencias expandidas — versión pro
Mejoras clave:
- CLI con argparse (CSV de entrada, columnas, idioma, batch size, modelos)
- Auto-detección de idioma para BERTScore o forzado vía --lang
- ROUGE-L y BLEU opcionales (evaluate)
- Cosine con SentenceTransformers (modelo configurable y device auto)
- Distinct-1/2 con tokenización simple/regex (ignora puntuación si se desea)
- Normalización opcional (lowercase, strip, quitar múltiple espacios)
- Exporta métricas por fila a CSV y resumen a JSON
- Bootstrap para IC al 95% (opcional)
- Reporte de “peores” ejemplos para depurar
"""

import argparse
import json
import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd

from evaluate import load as eval_load
from sentence_transformers import SentenceTransformer, util
import torch


def parse_args():
    ap = argparse.ArgumentParser(description="Evalúa salidas de un LLM contra referencias expandidas")
    ap.add_argument("--csv", default="validacion_llama3_with_expanded_refs.csv", help="Ruta CSV de entrada")
    ap.add_argument("--col_pred", default="prediccion_modelo", help="Columna con predicciones del LLM")
    ap.add_argument("--col_ref", default="referencia_expanded", help="Columna con referencias expandidas")
    ap.add_argument("--lang", default="auto", help="Idioma para BERTScore (auto|en|es|pt|...)")
    ap.add_argument("--do_rouge", action="store_true", help="Calcular ROUGE-L")
    ap.add_argument("--do_bleu", action="store_true", help="Calcular BLEU (sacrebleu)")
    ap.add_argument("--st_model", default="all-MiniLM-L6-v2", help="Modelo SentenceTransformers para cosine")
    ap.add_argument("--batch", type=int, default=64, help="Batch size para embeddings")
    ap.add_argument("--normalize", action="store_true", help="Normalizar textos (lower, espacios)")
    ap.add_argument("--strip_punct", action="store_true", help="Quitar puntuación al calcular distinct y longitudes")
    ap.add_argument("--bootstrap", type=int, default=0, help="N réplicas bootstrap para IC 95% (0=off)")
    ap.add_argument("--seed", type=int, default=42, help="Semilla de aleatoriedad")
    ap.add_argument("--out_prefix", default="eval_llm", help="Prefijo de archivos de salida")
    return ap.parse_args()


def normalize_text(s: str, strip_punct: bool, lower: bool = True) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.strip()
    if lower:
        s = s.lower()
    s = re.sub(r"\s+", " ", s)
    if strip_punct:
        # quita signos de puntuación y símbolos
        s = re.sub(r"[\p{P}\p{S}]", "", s)
    return s


def detect_lang_majority(texts: List[str]) -> str:
    # Heurística simple basada en caracteres del español
    es_markers = sum(any(ch in t for ch in "áéíóúñ¿¡") for t in texts)
    ratio = es_markers / max(1, len(texts))
    return "es" if ratio > 0.2 else "en"


def bertscore_scores(preds: List[str], refs: List[str], lang: str) -> Tuple[float, List[float]]:
    metric = eval_load("bertscore")
    res = metric.compute(predictions=preds, references=refs, lang=lang)
    f1 = np.array(res["f1"], dtype=float)
    return float(f1.mean()), f1.tolist()


def rouge_scores(preds: List[str], refs: List[str]) -> Tuple[float, List[float]]:
    rouge = eval_load("rouge")
    out = rouge.compute(predictions=preds, references=refs)
    per_item = []
    for p, r in zip(preds, refs):
        d = rouge.compute(predictions=[p], references=[r])
        per_item.append(d.get("rougeL", 0.0))
    return float(out.get("rougeL", 0.0)), per_item


def bleu_scores(preds: List[str], refs: List[str]) -> Tuple[float, List[float]]:
    bleu = eval_load("bleu")
    per_item = []
    for p, r in zip(preds, refs):
        d = bleu.compute(predictions=[p], references=[[r]])
        per_item.append(d.get("bleu", 0.0))
    global_mean = float(np.mean(per_item)) if per_item else 0.0
    return global_mean, per_item


def cosine_diag(preds: List[str], refs: List[str], st_model: str, batch: int) -> Tuple[float, List[float]]:
    model = SentenceTransformer(st_model, device="cuda" if torch.cuda.is_available() else "cpu")
    emb_ref = model.encode(refs, batch_size=batch, convert_to_tensor=True, normalize_embeddings=True)
    emb_pred = model.encode(preds, batch_size=batch, convert_to_tensor=True, normalize_embeddings=True)
    cosine = util.cos_sim(emb_pred, emb_ref).diagonal().detach().cpu().numpy().astype(float)
    return float(cosine.mean()), cosine.tolist()


def distinct_n(sentences: List[str], n: int = 1) -> float:
    ngrams_set = set()
    total_ngrams = 0
    for sent in sentences:
        tokens = sent.split()
        if len(tokens) < n:
            continue
        ngrams_sent = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        ngrams_set.update(ngrams_sent)
        total_ngrams += len(ngrams_sent)
    if total_ngrams == 0:
        return 0.0
    return len(ngrams_set) / total_ngrams


def bootstrap_ci(values: List[float], iters: int = 1000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = np.array(values, dtype=float)
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    means = []
    n = len(vals)
    for _ in range(iters):
        sample = vals[rng.integers(0, n, size=n)]
        means.append(sample.mean())
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)

    if not os.path.exists(args.csv):
        raise FileNotFoundError(args.csv)

    df = pd.read_csv(args.csv)
    if args.col_pred not in df.columns or args.col_ref not in df.columns:
        raise ValueError(f"El CSV debe contener '{args.col_pred}' y '{args.col_ref}'. Columnas: {list(df.columns)}")

    df_valid = df.dropna(subset=[args.col_pred, args.col_ref]).copy()

    # Normalización opcional
    if args.normalize or args.strip_punct:
        df_valid["_pred_nrm"] = df_valid[args.col_pred].apply(lambda s: normalize_text(s, args.strip_punct))
        df_valid["_ref_nrm"]  = df_valid[args.col_ref].apply(lambda s: normalize_text(s, args.strip_punct))
        preds = df_valid["_pred_nrm"].tolist()
        refs  = df_valid["_ref_nrm"].tolist()
    else:
        preds = df_valid[args.col_pred].astype(str).tolist()
        refs  = df_valid[args.col_ref].astype(str).tolist()

    # Idioma BERTScore
    lang = detect_lang_majority(refs) if args.lang == "auto" else args.lang

    print("=== Evaluación iniciada ===")
    print(f"Filas válidas: {len(df_valid)} | Idioma BERTScore: {lang}")

    # BERTScore
    print("\nCalculando BERTScore...")
    bert_mean, bert_per_item = bertscore_scores(preds, refs, lang=lang)
    print(f"BERTScore (F1 promedio): {bert_mean:.4f}")

    # Cosine
    print("\nCalculando similitud coseno (SentenceTransformers)...")
    cosine_mean, cosine_per_item = cosine_diag(preds, refs, args.st_model, args.batch)
    print(f"Cosine (diag) promedio: {cosine_mean:.4f}")

    # ROUGE / BLEU opcionales
    rouge_mean, rouge_per_item = (float("nan"), [])
    if args.do_rouge:
        print("\nCalculando ROUGE-L...")
        rouge_mean, rouge_per_item = rouge_scores(preds, refs)
        print(f"ROUGE-L promedio: {rouge_mean:.4f}")

    bleu_mean, bleu_per_item = (float("nan"), [])
    if args.do_bleu:
        print("\nCalculando BLEU...")
        bleu_mean, bleu_per_item = bleu_scores(preds, refs)
        print(f"BLEU promedio: {bleu_mean:.4f}")

    # Diversidad
    print("\nCalculando Distinct-1/2...")
    distinct1 = distinct_n(preds, n=1)
    distinct2 = distinct_n(preds, n=2)
    print(f"Distinct-1: {distinct1:.4f} | Distinct-2: {distinct2:.4f}")

    # Longitud
    lengths = [len(p.split()) for p in preds]
    mean_len = float(np.mean(lengths)) if lengths else float("nan")
    std_len  = float(np.std(lengths)) if lengths else float("nan")
    print(f"Longitud media: {mean_len:.1f} ± {std_len:.1f} palabras")

    # Bootstrap (IC 95%)
    ci = {}
    if args.bootstrap and len(df_valid) > 5:
        print("\nCalculando intervalos de confianza (bootstrap 95%)...")
        for name, per_item in {
            "bertscore_f1": bert_per_item,
            "cosine_diag": cosine_per_item,
            **({"rougeL": rouge_per_item} if args.do_rouge else {}),
            **({"bleu": bleu_per_item} if args.do_bleu else {}),
        }.items():
            lo, hi = bootstrap_ci(per_item, iters=args.bootstrap, seed=args.seed)
            ci[name] = {"lo": lo, "hi": hi}
            print(f"IC95% {name}: [{lo:.4f}, {hi:.4f}]")

    # Guardar por-fila
    df_valid["metric_bertscore_f1"] = bert_per_item
    df_valid["metric_cosine_diag"]  = cosine_per_item
    if args.do_rouge:
        df_valid["metric_rougeL"] = rouge_per_item
    if args.do_bleu:
        df_valid["metric_bleu"] = bleu_per_item
    df_valid["metric_len_words"]  = lengths

    out_csv = f"{args.out_prefix}_per_item.csv"
    df_valid.to_csv(out_csv, index=False)

    # Resumen
    summary = {
        "rows": int(len(df_valid)),
        "lang": lang,
        "bertscore_f1_mean": float(bert_mean),
        "cosine_diag_mean": float(cosine_mean),
        "rougeL_mean": float(rouge_mean),
        "bleu_mean": float(bleu_mean),
        "distinct1": float(distinct1),
        "distinct2": float(distinct2),
        "len_mean": float(mean_len),
        "len_std": float(std_len),
        "bootstrap_ci": ci,
        "config": vars(args),
    }

    out_json = f"{args.out_prefix}_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Reporte rápido: top-10 peores por BERTScore
    worst_idx = np.argsort(bert_per_item)[:10]
    print("\n=== Peores 10 por BERTScore ===")
    for i in worst_idx:
        try:
            print(f"[{i}] BERT={bert_per_item[i]:.4f} | COS={cosine_per_item[i]:.4f}")
            print(" PRED:", preds[i][:200])
            print(" REF :", refs[i][:200])
        except Exception:
            pass

    print("\nArchivos guardados:")
    print(" -", out_csv)
    print(" -", out_json)
    print("\nEvaluación completada ✅")
