# eval_ensemble_bert_roberta.py
import os, json, argparse
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification
)

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    f1_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -------------------- Config por defecto --------------------
DEFAULT_BERT_CKPT    = "bert-finetuned-epoch9-acc0.9783"
DEFAULT_ROBERTA_CKPT = "roberta-finetuned-epoch8-acc0.9706"

DEFAULT_VAL_XLSX  = "validation.xlsx"
DEFAULT_TEST_XLSX = "test.xlsx"
DEFAULT_ENCODER   = "label_encoder.pkl"

TEXT_COL  = "English"
CLASS_COL = "Classification_English"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
MAX_LEN = 64

# -------------------- Utilidades --------------------
class TextDS(Dataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self): return len(self.texts)
    def __getitem__(self, i): return {"text": str(self.texts[i])}

def tokenize_batch(tokenizer, texts):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]

@torch.no_grad()
def model_logits(model, tokenizer, dataloader):
    model.eval()
    outs = []
    for batch in dataloader:
        texts = batch["text"]
        ids, att = tokenize_batch(tokenizer, texts)
        ids, att = ids.to(DEVICE), att.to(DEVICE)
        logits = model(input_ids=ids, attention_mask=att).logits
        outs.append(logits.cpu())
    return torch.cat(outs, dim=0).numpy()

def evaluate_probs(y_true, probs):
    preds = probs.argmax(axis=1)
    return dict(
        acc=accuracy_score(y_true, preds),
        f1_macro=f1_score(y_true, preds, average="macro", zero_division=0),
        f1_weighted=f1_score(y_true, preds, average="weighted", zero_division=0),
        prec_w=precision_score(y_true, preds, average="weighted", zero_division=0),
        rec_w=recall_score(y_true, preds, average="weighted", zero_division=0)
    )

def ensure_label_encoder(df_fit, encoder_path, label_col=CLASS_COL):
    if os.path.isfile(encoder_path):
        return joblib.load(encoder_path)
    le = LabelEncoder().fit(df_fit[label_col].astype(str).tolist())
    joblib.dump(le, encoder_path)
    return le

def get_labels(df, le, maybe_encoded_col="y_encoded", label_col=CLASS_COL):
    if maybe_encoded_col in df.columns:
        return df[maybe_encoded_col].values
    return le.transform(df[label_col].astype(str).tolist())

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
    # abreviaturas legibles
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

# -------------------- Pipeline --------------------
def main():
    ap = argparse.ArgumentParser(description="Ensamble BERT+RoBERTa por promedio ponderado (calibrado en validación).")
    ap.add_argument("--bert_ckpt", default=DEFAULT_BERT_CKPT)
    ap.add_argument("--roberta_ckpt", default=DEFAULT_ROBERTA_CKPT)
    ap.add_argument("--val_xlsx", default=DEFAULT_VAL_XLSX)
    ap.add_argument("--test_xlsx", default=DEFAULT_TEST_XLSX)
    ap.add_argument("--label_encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--grid_step", type=float, default=0.05, help="Paso del barrido de pesos w en [0,1].")
    args = ap.parse_args()

    # --- Carga validación / test ---
    df_val  = pd.read_excel(args.val_xlsx)
    df_test = pd.read_excel(args.test_xlsx)

    # --- LabelEncoder ---
    le = ensure_label_encoder(df_val, args.label_encoder, label_col=CLASS_COL)
    class_names = list(le.classes_)
    y_val  = get_labels(df_val,  le)
    y_test = get_labels(df_test, le)

    texts_val  = df_val[TEXT_COL].astype(str).tolist()
    texts_test = df_test[TEXT_COL].astype(str).tolist()

    # --- Tokenizers y modelos ---
    bert_tok = None; bert = None
    rob_tok  = None; roberta = None

    print("Cargando BERT:", args.bert_ckpt)
    bert_tok = BertTokenizer.from_pretrained(args.bert_ckpt, local_files_only=True)
    bert     = BertForSequenceClassification.from_pretrained(args.bert_ckpt, local_files_only=True).to(DEVICE)

    print("Cargando RoBERTa:", args.roberta_ckpt)
    rob_tok  = RobertaTokenizer.from_pretrained(args.roberta_ckpt, local_files_only=True)
    roberta  = RobertaForSequenceClassification.from_pretrained(args.roberta_ckpt, local_files_only=True).to(DEVICE)

    # --- Dataloaders ---
    val_loader  = DataLoader(TextDS(texts_val),  batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TextDS(texts_test), batch_size=args.batch_size, shuffle=False)

    # --- Logits y probabilidades ---
    print("Inferencia en VALIDACIÓN...")
    logits_val_bert = model_logits(bert, bert_tok, val_loader)
    logits_val_rob  = model_logits(roberta, rob_tok, val_loader)
    probs_val_bert  = F.softmax(torch.tensor(logits_val_bert), dim=1).numpy()
    probs_val_rob   = F.softmax(torch.tensor(logits_val_rob ), dim=1).numpy()

    # --- Barrido de w para F1 macro ---
    grid = np.arange(0.0, 1.0 + 1e-9, args.grid_step)
    best = (-1.0, 0.5)
    for w in grid:
        probs_w = w*probs_val_bert + (1-w)*probs_val_rob
        score = evaluate_probs(y_val, probs_w)["f1_macro"]
        if score > best[0]:
            best = (score, w)
    BEST_W = float(best[1])
    print(f"Mejor w (F1 macro en validación): {BEST_W:.2f} -> {best[0]:.4f}")

    # --- Test ---
    print("Inferencia en TEST con w óptimo...")
    logits_test_bert = model_logits(bert, bert_tok, test_loader)
    logits_test_rob  = model_logits(roberta, rob_tok, test_loader)
    probs_test_bert  = F.softmax(torch.tensor(logits_test_bert), dim=1).numpy()
    probs_test_rob   = F.softmax(torch.tensor(logits_test_rob ), dim=1).numpy()

    probs_test = BEST_W*probs_test_bert + (1-BEST_W)*probs_test_rob
    report = evaluate_probs(y_test, probs_test)
    print("TEST ensamble:", report)

    # --- Salidas y artefactos ---
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = f"ensemble_outputs_{ts}"
    os.makedirs(out_dir, exist_ok=True)

    # Guardar config/resultados
    with open(os.path.join(out_dir, "ensemble_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "best_w": BEST_W,
            "val_macro_f1": float(best[0]),
            "test_metrics": {k: float(v) for k, v in report.items()},
            "bert_ckpt": args.bert_ckpt,
            "roberta_ckpt": args.roberta_ckpt,
            "val_file": args.val_xlsx,
            "test_file": args.test_xlsx,
            "label_encoder": args.label_encoder
        }, f, ensure_ascii=False, indent=2)

    # Predicciones y matriz por clase
    y_pred_test = probs_test.argmax(axis=1)
    df_metrics = metrics_table(y_test, y_pred_test, class_names)
    df_metrics.to_csv(os.path.join(out_dir, "metricas_por_clase_test_ensemble.csv"), index=True, encoding="utf-8-sig")
    try:
        df_metrics.to_excel(os.path.join(out_dir, "metricas_por_clase_test_ensemble.xlsx"), index=True)
    except Exception:
        pass

    # Matriz de confusión
    png = os.path.join(out_dir, "matriz_confusion_test_ensemble.png")
    plot_confusion_matrix(y_test, y_pred_test, class_names,
                          title=f"Confusion Matrix (Test) - Ensemble BERT({BEST_W:.2f}) + RoBERTa({1-BEST_W:.2f})",
                          outfile=png)
    print(f"Artefactos guardados en: {out_dir}")

if __name__ == "__main__":
    main()
