#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QLoRA LLaMA-3-8B-Instruct + métricas + gráficos SIN W&B
- eval_loss → perplexity
- ROUGE-L, BERTScore(F1), Cosine
- Guarda CSV y gráficos PNG por época/step (Matplotlib)
"""

import os, math, json, random
from typing import Dict, List, Any, Optional

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    BitsAndBytesConfig, set_seed, TrainerCallback
)
from peft import LoraConfig, get_peft_model
from evaluate import load as eval_load
from sentence_transformers import SentenceTransformer, util

# ========= CONFIG =========
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DATA_PATH  = "dataset_sft.jsonl"           # columnas: prompt, completion, (opcional) split
OUTPUT_DIR = "./qlora_llama3_metrics_adapter"

NUM_EPOCHS = 5
PER_DEVICE_BATCH_SIZE = 8
GRAD_ACCUM = 16
LR = 2e-4
MAX_LENGTH = 768
VAL_SIZE = 0.1
SEED = 42
EVAL_GENERATE_SAMPLES = 256   # pares para métricas de generación
LANG_BERTSCORE = "en"         # cambia a "es" si tus refs están en español

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========= 4-bit =========
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16
)

print("Cargando tokenizer/modelo...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=False,
    low_cpu_mem_usage=True
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

print("Cargando dataset...")
raw = load_dataset("json", data_files=DATA_PATH, split="train")
if "split" not in raw.column_names:
    raw = raw.train_test_split(test_size=VAL_SIZE, seed=SEED)
    ds_train, ds_eval = raw["train"], raw["test"]
else:
    ds_train = raw.filter(lambda r: r.get("split","train") == "train")
    ds_eval  = raw.filter(lambda r: r.get("split","train") != "train")

def tokenize_fn(examples):
    inputs, labels, masks = [], [], []
    for p, c in zip(examples["prompt"], examples["completion"]):
        p = p or ""; c = c or ""
        full = p + c
        enc = tokenizer(full, truncation=True, max_length=MAX_LENGTH, padding="max_length")
        prompt_len = len(tokenizer(p, truncation=True, max_length=MAX_LENGTH)["input_ids"])
        label_ids = enc["input_ids"].copy()
        for i in range(min(prompt_len, len(label_ids))):
            label_ids[i] = -100
        inputs.append(enc["input_ids"]); labels.append(label_ids); masks.append(enc["attention_mask"])
    return {"input_ids": inputs, "labels": labels, "attention_mask": masks}

print("Tokenizando...")
col_drop = [c for c in ds_train.column_names if c not in ("prompt","completion","split")]
tok_train = ds_train.map(tokenize_fn, batched=True, remove_columns=col_drop)
tok_eval  = ds_eval.map(tokenize_fn,  batched=True, remove_columns=col_drop)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ========= Métricas de generación =========
def _sample_eval_pairs(dataset: Dataset, k: int) -> List[Dict[str, str]]:
    idxs = list(range(len(dataset)))
    random.Random(SEED).shuffle(idxs)
    idxs = idxs[:min(k, len(dataset))]
    out = []
    for i in idxs:
        row = ds_eval[int(i)]
        out.append({"prompt": row["prompt"], "completion": row["completion"]})
    return out

def _generate_predictions(model, tokenizer, pairs: List[Dict[str, str]]) -> List[str]:
    preds = []
    device = next(model.parameters()).device
    for ex in pairs:
        msgs = [{"role":"user", "content": ex["prompt"]}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=True,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        gen_ids = out[0]
        input_len = enc["input_ids"].shape[1]
        new_tokens = gen_ids[input_len:]
        preds.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return preds

def compute_generation_metrics(model, tokenizer, eval_pairs: List[Dict[str,str]]) -> Dict[str, float]:
    if len(eval_pairs) == 0:
        return {}
    refs = [x["completion"] for x in eval_pairs]
    preds = _generate_predictions(model, tokenizer, eval_pairs)

    rouge = eval_load("rouge")
    rouge_out = rouge.compute(predictions=preds, references=refs)
    rougeL = float(rouge_out.get("rougeL", 0.0))

    bertscore = eval_load("bertscore")
    bs = bertscore.compute(predictions=preds, references=refs, lang=LANG_BERTSCORE)
    bert_f1 = float(np.mean(bs["f1"]))

    st = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")
    emb_ref  = st.encode(refs,  convert_to_tensor=True, normalize_embeddings=True)
    emb_pred = st.encode(preds, convert_to_tensor=True, normalize_embeddings=True)
    cosine_scores = util.cos_sim(emb_pred, emb_ref).diagonal().detach().cpu().numpy().astype(float)
    cosine_mean = float(np.mean(cosine_scores))

    return {"gen_rougeL": rougeL, "gen_bertscore_f1": bert_f1, "gen_cosine": cosine_mean}

# ========= Callback para recolectar logs y graficar =========
class MetricsHistory(TrainerCallback):
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.rows = []  # dicts con step/epoch/metrics

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None: return
        row = {"step": state.global_step, "epoch": state.epoch}
        row.update({k: float(v) for k,v in logs.items() if isinstance(v, (int, float))})
        self.rows.append(row)

    def on_train_end(self, args, state, control, **kwargs):
        if not self.rows: return
        df = pd.DataFrame(self.rows)
        csv_path = os.path.join(self.out_dir, "training_logs.csv")
        df.to_csv(csv_path, index=False)

        # Graficar (1 métrica por gráfico; sin estilos/colores explícitos)
        def _plot(x, y, title, fname):
            plt.figure()
            plt.plot(df[x], df[y])
            plt.xlabel(x); plt.ylabel(y); plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_dir, fname), dpi=150)
            plt.close()

        # Si existen columnas, graficamos
        if "loss" in df.columns:
            _plot("step", "loss", "Train loss por step", "train_loss_step.png")
        if "eval_loss" in df.columns:
            _plot("step", "eval_loss", "Eval loss por step", "eval_loss_step.png")
        if "perplexity" in df.columns:
            _plot("step", "perplexity", "Perplexity por step", "perplexity_step.png")
        if "gen_rougeL" in df.columns:
            _plot("step", "gen_rougeL", "ROUGE-L por step", "rougeL_step.png")
        if "gen_bertscore_f1" in df.columns:
            _plot("step", "gen_bertscore_f1", "BERTScore(F1) por step", "bertscore_step.png")
        if "gen_cosine" in df.columns:
            _plot("step", "gen_cosine", "Cosine similarity por step", "cosine_step.png")

# ========= Trainer con métricas de generación =========
class GenMetricsTrainer(Trainer):
    def evaluate(self, eval_dataset: Optional[Dataset] = None, **kwargs):
        base = super().evaluate(eval_dataset=eval_dataset, **kwargs)
        base["perplexity"] = float(math.exp(base["eval_loss"])) if "eval_loss" in base and base["eval_loss"] > 0 else float("nan")
        pairs = _sample_eval_pairs(ds_eval, k=EVAL_GENERATE_SAMPLES)
        try:
            gen = compute_generation_metrics(self.model, tokenizer, pairs)
            base.update(gen)
        except Exception as e:
            print("[WARN] compute_generation_metrics:", e)
        self.log(base)
        return base

# ========= TrainingArguments =========
set_seed(SEED)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=max(1, PER_DEVICE_BATCH_SIZE // 2),
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    bf16=torch.cuda.is_available(),
    fp16=False,
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",  # <- sin W&B/TensorBoard
)

history_cb = MetricsHistory(OUTPUT_DIR)

trainer = GenMetricsTrainer(
    model=model,
    args=training_args,
    train_dataset=tok_train,
    eval_dataset=tok_eval,
    data_collator=data_collator,
    callbacks=[history_cb],
)

# ========= Train & Eval =========
print("Entrenando...")
trainer.train()

print("Evaluando final...")
final_metrics = trainer.evaluate()
print("\n=== MÉTRICAS FINALES ===")
for k, v in final_metrics.items():
    print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

# ========= Guardar adapter y métricas =========
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, "final_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(final_metrics, f, ensure_ascii=False, indent=2)

print("\nListo ✅")
print("Archivos generados en:", OUTPUT_DIR)
print("- training_logs.csv (todas las métricas por step)")
print("- *step.png (gráficos PNG por métrica)")
print("- final_metrics.json (resumen)")
