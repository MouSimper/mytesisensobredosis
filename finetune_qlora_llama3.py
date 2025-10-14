# finetune_qlora_llama3_test.py
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# --------------- CONFIG ----------------
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DATA_PATH = "dataset_sft.jsonl"
OUTPUT_DIR = "./qlora_llama3_test_adapter"

# Parámetros para prueba rápida
NUM_EPOCHS = 1             # 1 epoch
PER_DEVICE_BATCH_SIZE = 1   # batch pequeño
GRAD_ACCUM = 4             # simula batch más grande
LR = 2e-4
MAX_LENGTH = 256            # tokens más cortos
SAMPLE_SIZE = 200           # usar solo 200 ejemplos para prueba
# ----------------------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Loading tokenizer and 4-bit base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
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
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Cargar dataset y tomar solo SAMPLE_SIZE ejemplos
ds = load_dataset("json", data_files=DATA_PATH, split="train")
ds = ds.select(range(min(SAMPLE_SIZE, len(ds))))

def tokenize_fn(examples):
    inputs, labels, masks = [], [], []
    for p, c in zip(examples["prompt"], examples["completion"]):
        full = p + c
        enc = tokenizer(full, truncation=True, max_length=MAX_LENGTH, padding="max_length")
        prompt_len = len(tokenizer(p)["input_ids"])
        label_ids = enc["input_ids"].copy()
        for i in range(prompt_len):
            label_ids[i] = -100
        inputs.append(enc["input_ids"])
        labels.append(label_ids)
        masks.append([1]*MAX_LENGTH)
    return {"input_ids": inputs, "labels": labels, "attention_mask": masks}

tokenized = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    bf16=True,
    fp16=False,
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=1,
    remove_unused_columns=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

print("Start training (test run)...")
trainer.train()
print("Saving LoRA adapter...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done. Adapter saved in", OUTPUT_DIR)
