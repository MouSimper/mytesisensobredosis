# finetune_qlora_llama3_test_with_loss_curve.py
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, BitsAndBytesConfig, TrainerCallback
)
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt

# --------------- CONFIG ----------------
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DATA_PATH = "dataset_sft.jsonl"
OUTPUT_DIR = "./qlora_llama3_test_adapter"

NUM_EPOCHS = 5
PER_DEVICE_BATCH_SIZE = 4
GRAD_ACCUM = 8
LR = 1e-4
MAX_LENGTH = 512
SAMPLE_SIZE = None
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
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Cargar dataset
ds = load_dataset("json", data_files=DATA_PATH, split="train")
if SAMPLE_SIZE is not None:
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

# Callback para guardar el loss de cada step
class LossLoggerCallback(TrainerCallback):
    def __init__(self):
        self.step_losses = []
        self.epoch_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.step_losses.append(logs["loss"])

    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch_losses.append(self.step_losses.copy())
        # Graficar el loss al final de la época
        plt.figure(figsize=(8,5))
        plt.plot(self.step_losses, label="Train loss")
        plt.title(f"Training Loss - Epoch {state.epoch:.0f}")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(OUTPUT_DIR, f"train_loss_epoch_{int(state.epoch)}.png"))
        plt.close()
        self.step_losses = []

loss_logger = LossLoggerCallback()

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
    save_steps=200,
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
    callbacks=[loss_logger]
)

print("Start training...")
trainer.train()

print("Saving LoRA adapter...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done. Adapter saved in", OUTPUT_DIR)
