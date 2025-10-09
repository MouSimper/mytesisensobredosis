# train_eval_bert.py
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
from tqdm import tqdm

from transformers import (
    BertTokenizer, BertForSequenceClassification, get_scheduler
)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    f1_score, accuracy_score
)

# -------------------- Config --------------------
SEED = 42
EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
PATIENCE = 2
MAX_LEN = 64
MODEL_NAME = "bert-base-uncased" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    """
    Matriz de confusión con abreviaturas en ejes y leyenda a la derecha.
    max_len: longitud máxima de la abreviatura generada por clase.
    """
    from matplotlib import pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # --- generar abreviaturas únicas y legibles ---
    stop = {"of","the","and","for","to","in","on","a","an","de","la","el","los","las","y"}
    used = set()
    short_labels = []
    for lab in class_names:
        tokens = [t for t in lab.replace("/", " ").replace("-", " ").split() if t]
        code = "".join([t[0].upper() for t in tokens if t.lower() not in stop])[:max_len]
        if not code:
            code = "".join(lab.upper().split())[:max_len] or "C"
        base = code; k = 1
        while code in used:
            k += 1
            suf = str(k)
            code = (base[:max_len - len(suf)] + suf) if len(base) >= len(suf) else (base + suf)
        used.add(code)
        short_labels.append(code)

    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(
        cm,
        index=[f"R:{s}" for s in short_labels],
        columns=[f"P:{s}" for s in short_labels]
    )

    plt.figure(figsize=(12, 9))
    ax = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(title)

    legend_text = "\n".join(f"{s} = {full}" for s, full in zip(short_labels, class_names))
    plt.tight_layout(rect=[0.0, 0.0, 0.80, 1.0])
    plt.gcf().text(0.825, 0.5, legend_text, va="center", fontsize=9)

    plt.savefig(outfile, dpi=200)
    plt.close()

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

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train(); total_loss = 0.0
    loop = tqdm(dataloader, desc="Training", leave=False)
    for batch in loop:
        optimizer.zero_grad()
        ids = batch["input_ids"].to(device)
        att = batch["attention_mask"].to(device)
        lab = batch["labels"].to(device)
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        loss.backward(); optimizer.step(); scheduler.step()
        total_loss += loss.item(); loop.set_postfix(loss=float(loss.item()))
    return total_loss / max(1, len(dataloader))

def main():
    set_seed()

    # --- Carga dataset (mismo archivo que usas con RoBERTa) ---
    df = pd.read_excel("dataset_balanceado_limpio.xlsx")
    texts = df["English"].astype(str).tolist()
    labels_txt = df["Classification_English"].astype(str).tolist()

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels_txt)
    joblib.dump(label_encoder, "label_encoder.pkl")  # reutilizable para ensamble

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_ds = TextDataset(train_texts, train_labels, tokenizer)
    val_ds   = TextDataset(val_texts,   val_labels,   tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_encoder.classes_)
    ).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    steps = EPOCHS * len(train_loader)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=steps)

    best_val_acc = 0.0; counter = 0; best_model_path = ""
    tr_losses, v_accs, v_f1s, v_precs, v_recs = [], [], [], [], []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        tr_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        print(f"Train Loss: {tr_loss:.4f}")

        model.eval(); vp, yt = [], []
        with torch.no_grad():
            for b in val_loader:
                ids = b["input_ids"].to(DEVICE)
                att = b["attention_mask"].to(DEVICE)
                lab = b["labels"].to(DEVICE)
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

        tr_losses.append(tr_loss); v_accs.append(acc)
        v_f1s.append(f1w); v_precs.append(prec); v_recs.append(rec)

        if acc > best_val_acc:
            best_val_acc = acc; counter = 0
            best_model_path = f"./bert-finetuned-epoch{epoch+1}-acc{acc:.4f}"
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"Mejor modelo guardado en: {best_model_path}")
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            plot_confusion_matrix(yt, vp, label_encoder.classes_,
                                  title=f"Matriz de Validación (Best Epoch {epoch+1}) - BERT",
                                  outfile=f"matriz_confusion_valid_best_bert_{ts}.png")
            df_val.to_csv(f"metricas_valid_best_bert_{ts}.csv", index=True, encoding="utf-8-sig")
        else:
            counter += 1
            if counter >= PATIENCE:
                print("Early stopping activado."); break

    # Curvas
    ep = range(1, len(tr_losses)+1)
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1); plt.plot(ep, tr_losses, marker='o', label="Train Loss")
    plt.title("Curva de Pérdida (BERT)"); plt.xlabel("Época"); plt.ylabel("Loss"); plt.grid(); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(ep, v_accs, marker='o', label="Accuracy")
    plt.plot(ep, v_f1s, marker='s', label="F1-score (weighted)")
    plt.plot(ep, v_precs, marker='^', label="Precision (weighted)")
    plt.plot(ep, v_recs, marker='v', label="Recall (weighted)")
    plt.title("Curvas de Validación (BERT)")
    plt.xlabel("Época"); plt.ylabel("Score"); plt.grid(); plt.legend()

    plt.tight_layout(); plt.savefig("curvas_entrenamiento_bert.png", dpi=200); plt.show()

    # Test
    print("\nEvaluando el mejor modelo en TEST...")
    df_test = pd.read_excel("test.xlsx")
    t_texts = df_test["English"].astype(str).tolist()
    t_labels = label_encoder.transform(df_test["Classification_English"].astype(str).tolist())

    test_ds = TextDataset(t_texts, t_labels, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    best_model = BertForSequenceClassification.from_pretrained(best_model_path).to(DEVICE)
    best_model.eval(); yp, yt = [], []
    with torch.no_grad():
        for b in test_loader:
            ids = b["input_ids"].to(DEVICE)
            att = b["attention_mask"].to(DEVICE)
            lab = b["labels"].to(DEVICE)
            logits = best_model(input_ids=ids, attention_mask=att).logits
            yp.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            yt.extend(lab.cpu().numpy())

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    png = f"matriz_confusion_test_bert_{ts}.png"
    plot_confusion_matrix(yt, yp, label_encoder.classes_,
                          title="Matriz de Confusión (Test) - BERT", outfile=png)
    print(f"Matriz de confusión guardada: {png}")

    dfm = metrics_table(yt, yp, label_encoder.classes_)
    dfm.to_csv(f"metricas_por_clase_test_bert_{ts}.csv", index=True, encoding="utf-8-sig")
    try: dfm.to_excel(f"metricas_por_clase_test_bert_{ts}.xlsx", index=True)
    except Exception: pass
    print("Resumen (macro/weighted):")
    print(dfm.loc[["Macro avg","Weighted avg"]])

if __name__ == "__main__":
    main()
