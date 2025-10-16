#!/usr/bin/env python3
# train_eval_roberta_3050.py
# RoBERTa (base) afinado para 6GB VRAM (RTX 3050) con AMP + grad. accumulation

import os, random
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast
from torch.amp import GradScaler
from tqdm import tqdm

from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification, get_scheduler
)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    f1_score, accuracy_score
)

# -------------------- Config (ajustada a 6GB) --------------------
SEED = 42
EPOCHS = 8
BATCH_SIZE = 8                 # pequeño para 6GB
GRAD_ACCUM_STEPS = 4           # 8 x 4 = 32 (batch efectivo)
LEARNING_RATE = 3e-5
PATIENCE = 3
MAX_LEN = 128
MODEL_NAME = "roberta-base"    # liviano y fiable
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()
scaler = GradScaler(enabled=USE_CUDA)

# -------------------- Utilidades --------------------
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def metrics_table(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class    = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class        = f1_score(y_true, y_pred, average=None, zero_division=0)
    acc_per_class       = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    df = pd.DataFrame({
        "Clase": class_names,
        "Precision": precision_per_class,
        "Recall": recall_per_class,
        "F1": f1_per_class,
        "Accuracy": acc_per_class
    })
    df.loc["Weighted avg"] = [
        "Weighted avg",
        precision_score(y_true, y_pred, average="weighted", zero_division=0),
        recall_score(y_true, y_pred, average="weighted", zero_division=0),
        f1_score(y_true, y_pred, average="weighted", zero_division=0),
        accuracy_score(y_true, y_pred)
    ]
    df.loc["Macro avg"] = [
        "Macro avg",
        precision_score(y_true, y_pred, average="macro", zero_division=0),
        recall_score(y_true, y_pred, average="macro", zero_division=0),
        f1_score(y_true, y_pred, average="macro", zero_division=0),
        accuracy_score(y_true, y_pred)
    ]
    return df

def plot_confusion_matrix(y_true, y_pred, class_names, title, outfile, max_len=6):
    stop = {"of","the","and","for","to","in","on","a","an","de","la","el","los","las","y"}
    used = set(); short_labels = []
    for lab in class_names:
        tokens = [t for t in lab.replace("/", " ").replace("-", " ").split() if t]
        code = "".join([t[0].upper() for t in tokens if t.lower() not in stop])[:max_len]
        if not code: code = "".join(lab.upper().split())[:max_len] or "C"
        base = code; k = 1
        while code in used:
            k += 1; suf = str(k)
            code = (base[:max_len - len(suf)] + suf) if len(base) >= len(suf) else (base + suf)
        used.add(code); short_labels.append(code)

    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=[f"R:{s}" for s in short_labels], columns=[f"P:{s}" for s in short_labels])

    plt.figure(figsize=(12, 9))
    ax = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(title)

    legend_text = "\n".join(f"{s} = {full}" for s, full in zip(short_labels, class_names))
    plt.tight_layout(rect=[0.0, 0.0, 0.80, 1.0])
    plt.gcf().text(0.825, 0.5, legend_text, va="center", fontsize=9)
    plt.savefig(outfile, dpi=200); plt.close()

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = texts; self.labels = labels
        self.tokenizer = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        text = str(self.texts[idx]); label = int(self.labels[idx])
        enc = self.tokenizer(text, max_length=self.max_len, padding="max_length",
                             truncation=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": torch.tensor(label, dtype=torch.long)
        }

# -------------------- Entrenamiento con acumulación --------------------
def train_epoch(model, dataloader, optimizer, scheduler, device, scaler, grad_accum_steps=GRAD_ACCUM_STEPS):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    loop = tqdm(dataloader, desc="Training", leave=False)
    for step, batch in enumerate(loop, start=1):
        ids = batch["input_ids"].to(device, non_blocking=True)
        att = batch["attention_mask"].to(device, non_blocking=True)
        lab = batch["labels"].to(device, non_blocking=True)

        with autocast(enabled=USE_CUDA):
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            loss = loss / grad_accum_steps  # esencial para acumulación

        if USE_CUDA:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % grad_accum_steps == 0:
            if USE_CUDA:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item() * grad_accum_steps  # deshacer la división para reportar

        loop.set_postfix(loss=float(total_loss / step))
    return total_loss / max(1, len(dataloader))

# -------------------- Main --------------------
def main():
    set_seed()
    if USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # ====== Carga dataset ======
    df = pd.read_excel("balanced_t5.xlsx")
    if not {"English_clean", "Classification_English"}.issubset(df.columns):
        raise ValueError("❌ El dataset debe contener 'English_clean' y 'Classification_English'.")

    texts = df["English_clean"].astype(str).tolist()
    labels_txt = df["Classification_English"].astype(str).tolist()

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels_txt)
    joblib.dump(label_encoder, "label_encoder.pkl")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    train_ds = TextDataset(train_texts, train_labels, tokenizer)
    val_ds   = TextDataset(val_texts,   val_labels,   tokenizer)

    loader_kwargs = dict(batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=USE_CUDA)
    train_loader = DataLoader(train_ds, **loader_kwargs)
    val_loader   = DataLoader(val_ds,  **{**loader_kwargs, "shuffle": False})

    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_encoder.classes_)
    ).to(DEVICE)

    # Ahorro extra de VRAM
    model.gradient_checkpointing_enable()

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    # Scheduler por actualización (no por batch) -> steps = EPOCHS * (len(train_loader)/accum)
    update_steps_per_epoch = int(np.ceil(len(train_loader) / GRAD_ACCUM_STEPS))
    total_updates = EPOCHS * update_steps_per_epoch
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_updates)

    best_val_acc = 0.0; counter = 0; best_model_path = ""
    tr_losses, v_accs, v_f1s, v_precs, v_recs = [], [], [], [], []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        tr_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE, scaler, GRAD_ACCUM_STEPS)
        print(f"Train Loss: {tr_loss:.4f}")

        # ====== Validación ======
        model.eval(); vp, yt = [], []
        with torch.no_grad():
            for b in val_loader:
                ids = b["input_ids"].to(DEVICE, non_blocking=True)
                att = b["attention_mask"].to(DEVICE, non_blocking=True)
                lab = b["labels"].to(DEVICE, non_blocking=True)
                with autocast(enabled=USE_CUDA):
                    logits = model(input_ids=ids, attention_mask=att).logits
                vp.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                yt.extend(lab.cpu().numpy())

        acc  = accuracy_score(yt, vp)
        f1w  = f1_score(yt, vp, average="weighted", zero_division=0)
        prec = precision_score(yt, vp, average="weighted", zero_division=0)
        rec  = recall_score(yt, vp, average="weighted", zero_division=0)
        print(f"Validation Accuracy: {acc:.4f}")

        df_val = metrics_table(yt, vp, label_encoder.classes_)
        print(df_val)

        tr_losses.append(tr_loss); v_accs.append(acc); v_f1s.append(f1w); v_precs.append(prec); v_recs.append(rec)

        if acc > best_val_acc:
            best_val_acc = acc; counter = 0
            best_model_path = f"./roberta-base-finetuned-epoch{epoch+1}-acc{acc:.4f}"
            os.makedirs(best_model_path, exist_ok=True)
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"✅ Mejor modelo guardado en: {best_model_path}")

            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            plot_confusion_matrix(yt, vp, label_encoder.classes_,
                                  title=f"Matriz de Validación (Best Epoch {epoch+1}) - RoBERTa-base",
                                  outfile=f"matriz_confusion_valid_best_roberta_{ts}.png")
            df_val.to_csv(f"metricas_valid_best_roberta_{ts}.csv", index=True, encoding="utf-8-sig")
        else:
            counter += 1
            if counter >= PATIENCE:
                print("⏹️ Early stopping activado."); break

    # ====== Curvas ======
    ep = range(1, len(tr_losses)+1)
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1); plt.plot(ep, tr_losses, marker='o', label="Train Loss")
    plt.title("Curva de Pérdida (RoBERTa-base)"); plt.xlabel("Época"); plt.ylabel("Loss"); plt.grid(); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(ep, v_accs, marker='o', label="Accuracy")
    plt.plot(ep, v_f1s, marker='s', label="F1-score (weighted)")
    plt.plot(ep, v_precs, marker='^', label="Precision (weighted)")
    plt.plot(ep, v_recs, marker='v', label="Recall (weighted)")
    plt.title("Curvas de Validación (RoBERTa-base)")
    plt.xlabel("Época"); plt.ylabel("Score"); plt.grid(); plt.legend()
    plt.tight_layout(); plt.savefig("curvas_entrenamiento_roberta.png", dpi=200); plt.show()

    # ====== TEST ======
    print("\nEvaluando el mejor modelo en TEST...")
    df_test = pd.read_excel("test_limpio.xlsx")

    # Acepta 'English' o 'English_clean'
    text_col = "English" if "English" in df_test.columns else "English_clean"
    if not {text_col, "Classification_English"}.issubset(df_test.columns):
        raise ValueError("❌ El test debe tener 'English' (o 'English_clean') y 'Classification_English'.")

    t_texts = df_test[text_col].astype(str).tolist()
    t_labels = label_encoder.transform(df_test["Classification_English"].astype(str).tolist())

    test_ds = TextDataset(t_texts, t_labels, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=USE_CUDA)

    best_model = RobertaForSequenceClassification.from_pretrained(best_model_path).to(DEVICE)
    best_model.eval(); yp, yt = [], []
    with torch.no_grad():
        for b in test_loader:
            ids = b["input_ids"].to(DEVICE, non_blocking=True)
            att = b["attention_mask"].to(DEVICE, non_blocking=True)
            lab = b["labels"].to(DEVICE, non_blocking=True)
            with autocast(enabled=USE_CUDA):
                logits = best_model(input_ids=ids, attention_mask=att).logits
            yp.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            yt.extend(lab.cpu().numpy())

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    png = f"matriz_confusion_test_roberta_{ts}.png"
    plot_confusion_matrix(yt, yp, label_encoder.classes_,
                          title="Matriz de Confusión (Test) - RoBERTa-base", outfile=png)
    print(f"Matriz de confusión guardada: {png}")

    dfm = metrics_table(yt, yp, label_encoder.classes_)
    dfm.to_csv(f"metricas_por_clase_test_roberta_{ts}.csv", index=True, encoding="utf-8-sig")
    try: dfm.to_excel(f"metricas_por_clase_test_roberta_{ts}.xlsx", index=True)
    except Exception: pass
    print("Resumen (macro/weighted):"); print(dfm.loc[["Macro avg","Weighted avg"]])

if __name__ == "__main__":
    main()
