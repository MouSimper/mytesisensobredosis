import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
dataset_path = "eval.jsonl"  # ⚠️ Ajusta si está en otra carpeta

# ============================================================
# 2. CARGA DEL MODELO Y TOKENIZER
# ============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)

# ============================================================
# 3. CONFIGURACIÓN DE LORA
# ============================================================
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# ============================================================
# 4. CARGA Y PREPROCESAMIENTO DEL DATASET
# ============================================================
dataset = load_dataset("json", data_files=dataset_path, split="train")

def preprocess_function(examples):
    texts = []
    for prompt, answer in zip(examples["prompt"], examples["expected_answer"]):
        # Puedes quitar las etiquetas si tu dataset ya está en formato instructivo
        text = f"<s>[INST] {prompt} [/INST] {answer}</s>"
        texts.append(text)
    model_inputs = tokenizer(texts, truncation=True, padding="max_length", max_length=128)
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)
train_dataset = tokenized_dataset

# ============================================================
# 5. CONFIGURACIÓN DE ENTRENAMIENTO
# ============================================================
training_args = TrainingArguments(
    output_dir="./results_qlora_llama3",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=10,  # 👈 10 épocas
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    gradient_checkpointing=True,
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ============================================================
# 6. ENTRENAMIENTO
# ============================================================
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    data_collator=data_collator,
)

trainer.train()

# ============================================================
# 7. GRÁFICO DE PÉRDIDA POR ÉPOCA
# ============================================================
log_history = trainer.state.log_history
df = pd.DataFrame(log_history)
df_loss = df[df["loss"].notnull()]

steps_per_epoch = len(train_dataset) // (
    training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
)
df_loss["epoch"] = df_loss["step"] / steps_per_epoch

plt.figure(figsize=(8, 5))
plt.plot(df_loss["epoch"], df_loss["loss"], marker="o", linestyle="-", color="blue")
plt.xlabel("Época")
plt.ylabel("Pérdida (Loss)")
plt.title("Curva de Entrenamiento - Llama 3 8B QLoRA")
plt.grid(True)
plt.tight_layout()
plt.show()
