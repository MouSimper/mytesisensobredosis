#!/usr/bin/env python3
# chatbot_llama3_8bit_offload_con_clasificador.py
# LLaMA-3 8B en int8 + offload CPU (bnb) + clasificador BERT+RoBERTa
# Carga en 2 pasos: auto → fijar embed/norm/lm_head en GPU si hace falta.
# Gradio Chat (type="messages").

import os, sys, traceback
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import gradio as gr

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification
)
from huggingface_hub import login as hf_login, HfFolder
import joblib

# =========================
# CONFIG LLaMA / GENERACIÓN
# =========================
MODEL_NAME         = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_NEW_TOKENS     = 256
TEMPERATURE        = 0.7
TOP_P              = 0.9
TOP_K              = 40
REPETITION_PENALTY = 1.1

# =========================
# TOKEN HF (HARDCODEADO)
# =========================
HF_TOKEN_HARDCODED = "REMOVED_TOKEN".strip()
if HF_TOKEN_HARDCODED:
    os.environ["HF_TOKEN"] = HF_TOKEN_HARDCODED
HF_TOKEN = os.environ.get("HF_TOKEN") or HfFolder.get_token() or HF_TOKEN_HARDCODED

# ==============
# Login a HF Hub
# ==============
if HF_TOKEN:
    try:
        hf_login(HF_TOKEN)
        HfFolder.save_token(HF_TOKEN)
        print("Hugging Face: autenticado con token embebido.")
    except Exception as e:
        print("Advertencia HF:", e)
else:
    print("⚠️ No se detectó HF_TOKEN (si no está cacheado, fallará).")

# ================
# Utilidades BnB/DM
# ================
def _normalize_device(dev):
    if isinstance(dev, int):
        return f"cuda:{dev}"
    if isinstance(dev, str):
        if dev in ("cpu", "disk", "meta"):
            return dev
        if dev.startswith("cuda"):
            return dev
    return "cpu"

def _pick_input_device_for_sharded_model(model):
    dm = getattr(model, "hf_device_map", None)
    if isinstance(dm, dict):
        # Prioriza embeddings
        for name, dev in dm.items():
            if "embed_tokens" in name:
                nd = _normalize_device(dev)
                return "cuda:0" if nd == "0" else nd
        # Si no, el primer CUDA que haya
        for dev in dm.values():
            nd = _normalize_device(dev)
            if nd.startswith("cuda"):
                return nd
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def _reload_with_pinned_heads(auto_map, common_kwargs, bnb_int8, offload_dir):
    """Devuelve un modelo recargado con embed/norm/lm_head fijados en cuda:0."""
    device_map = dict(auto_map)  # copia del mapa auto
    device_map["model.embed_tokens"] = 0
    device_map["model.norm"] = 0
    device_map["lm_head"] = 0
    # Asegura que no haya 'auto' dentro del dict
    for k, v in list(device_map.items()):
        if isinstance(v, str) and v == "auto":
            device_map[k] = "cpu"

    print("Recargando con device_map corregido (pinning embed/norm/lm_head en cuda:0)...")
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_int8,
        device_map=device_map,
        offload_folder=offload_dir,
        **common_kwargs
    )

# ===========================
# Cargar tokenizer y el modelo
# ===========================
print("Cargando tokenizer ...")
try:
    llama_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, token=HF_TOKEN, use_fast=True
    )
    if llama_tokenizer.pad_token_id is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token
except Exception as e:
    print("Error cargando tokenizer:", e)
    sys.exit(1)

print("Cargando modelo ...")
try:
    common_kwargs = dict(
        token=HF_TOKEN,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    if torch.cuda.is_available():
        bnb_int8 = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        offload_dir = "./offload_llama3_8bit"
        os.makedirs(offload_dir, exist_ok=True)

        # Paso 1: carga con auto
        llama_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_int8,
            device_map="auto",
            offload_folder=offload_dir,
            **common_kwargs
        )
        dm = getattr(llama_model, "hf_device_map", {})
        print("device_map (auto):", dm)

        # Paso 2: si embed/norm/lm_head no están en CUDA → recargar fijándolos
        need_reload = True
        for key in ("model.embed_tokens", "model.norm", "lm_head"):
            dev = _normalize_device(dm.get(key, "cpu"))
            if not dev.startswith("cuda"):
                need_reload = True
                break
        else:
            need_reload = False

        if need_reload:
            del llama_model
            torch.cuda.empty_cache()
            llama_model = _reload_with_pinned_heads(dm, common_kwargs, bnb_int8, offload_dir)
            print("LLaMA-3 8B (int8+offload) con embed/norm/lm_head en GPU (recargado).")
        else:
            print("LLaMA-3 8B (int8+offload) con mapa auto OK.")
    else:
        llama_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            **common_kwargs
        )
        print("LLaMA-3 8B cargado en CPU (fp32).")
    llama_model.eval()
except Exception as e:
    print("Error al cargar el modelo:\n")
    traceback.print_exc()
    sys.exit(1)

# ==================================
# CLASIFICADOR HÍBRIDO (BERT+RoBERTa)
# ==================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = []
bert = roberta = bert_tok = rob_tok = None

# ===============================
# CONFIG Clasificador Híbrido NLP
# ===============================
BERT_CKPT     = "bert-finetuned-epoch9-acc0.9783"
ROBERTA_CKPT  = "roberta-finetuned-epoch8-acc0.9706"
LABEL_ENCODER = "label_encoder.pkl"
MAX_LEN_CLF   = 64
BEST_W        = 0.50

def _try_load_classifier():
    global bert, roberta, bert_tok, rob_tok, CLASS_NAMES
    try:
        bert_tok = BertTokenizer.from_pretrained(BERT_CKPT, local_files_only=True)
        bert = BertForSequenceClassification.from_pretrained(BERT_CKPT, local_files_only=True).to(DEVICE)
        rob_tok = RobertaTokenizer.from_pretrained(ROBERTA_CKPT, local_files_only=True)
        roberta = RobertaForSequenceClassification.from_pretrained(ROBERTA_CKPT, local_files_only=True).to(DEVICE)
        label_encoder = joblib.load(LABEL_ENCODER)
        CLASS_NAMES = list(label_encoder.classes_)
        print("Clasificador híbrido cargado correctamente (local).")
        return True
    except Exception as e_local:
        print("Intento de carga local falló:", e_local)
        try:
            bert_tok = BertTokenizer.from_pretrained(BERT_CKPT, token=HF_TOKEN)
            bert = BertForSequenceClassification.from_pretrained(BERT_CKPT, token=HF_TOKEN).to(DEVICE)
            rob_tok = RobertaTokenizer.from_pretrained(ROBERTA_CKPT, token=HF_TOKEN)
            roberta = RobertaForSequenceClassification.from_pretrained(ROBERTA_CKPT, token=HF_TOKEN).to(DEVICE)
            label_encoder = joblib.load(LABEL_ENCODER)
            CLASS_NAMES = list(label_encoder.classes_)
            print("Clasificador híbrido cargado correctamente (descargado).")
            return True
        except Exception as e_remote:
            print("Error cargando el clasificador híbrido:", e_remote)
            print("Verifica rutas BERT_CKPT / ROBERTA_CKPT y LABEL_ENCODER.")
            return False

_classifier_ok = _try_load_classifier()

@torch.no_grad()
def classify_text_hybrid(text: str, best_w: float = BEST_W, topk: int = 3):
    if not _classifier_ok or not (bert and roberta and bert_tok and rob_tok and CLASS_NAMES):
        return {"error": "Clasificador no cargado. Revisa checkpoints y label_encoder.pkl."}
    eb = bert_tok(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN_CLF).to(DEVICE)
    er =  rob_tok(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN_CLF).to(DEVICE)
    pb = F.softmax(bert(**eb).logits, dim=1)
    pr = F.softmax(roberta(**er).logits, dim=1)
    probs = (best_w * pb + (1 - best_w) * pr)[0].detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    top_idx = np.argsort(probs)[::-1][:max(1, topk)]
    return {
        "label": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx] * 100),
        "top": [(CLASS_NAMES[i], float(probs[i] * 100)) for i in top_idx],
    }

# =========================================
# GENERACIÓN (evita meta tensor + warnings)
# =========================================
@torch.no_grad()
def llama_chat_generate(user_message: str, history: list | None = None) -> str:
    history = history or []
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for u, a in history:
        messages += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    messages.append({"role": "user", "content": user_message})

    prompt = llama_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = llama_tokenizer(prompt, return_tensors="pt")

    target_device = _pick_input_device_for_sharded_model(llama_model)
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    out = llama_model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        repetition_penalty=REPETITION_PENALTY,
        pad_token_id=llama_tokenizer.pad_token_id,
        eos_token_id=llama_tokenizer.eos_token_id,
    )

    gen_ids = out[0]
    input_len = inputs["input_ids"].shape[1]
    new_tokens = gen_ids[input_len:]
    return llama_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# =========================
# GRADIO: Chat + Clasificador
# =========================
def gradio_chat_fn(message, chat_history):
    pairs, last_user = [], None
    for m in chat_history or []:
        if m.get("role") == "user":
            last_user = m.get("content", "")
        elif m.get("role") == "assistant" and last_user is not None:
            pairs.append((last_user, m.get("content", "")))
            last_user = None

    msg = (message or "").strip()
    if not msg:
        return chat_history or [], ""

    if msg.lower().startswith("/clasificar"):
        text = msg.split("/clasificar", 1)[-1].strip(": ").strip()
        if not text:
            reply = "Usa: /clasificar <descripción del defecto>"
        else:
            res = classify_text_hybrid(text)
            reply = f"⚠️ {res['error']}" if "error" in res else (
                f"**Predicción:** {res['label']} ({res['confidence']:.2f}%)\n" +
                "Top probabilidades:\n" + "\n".join([f"- {lbl}: {p:.2f}%" for lbl, p in res["top"]])
            )
        chat_history = (chat_history or []) + [{"role":"user","content":message},{"role":"assistant","content":reply}]
        return chat_history, ""

    try:
        reply = llama_chat_generate(msg, pairs)
    except Exception as e:
        reply = f"⚠️ Error generando respuesta: {e}"

    chat_history = (chat_history or []) + [{"role":"user","content":message},{"role":"assistant","content":reply}]
    return chat_history, ""

def gradio_classify_fn(text):
    if not text or not text.strip():
        return "Ingrese una descripción para clasificar.", ""
    res = classify_text_hybrid(text.strip())
    if "error" in res:
        return f"⚠️ {res['error']}", ""
    table = pd.DataFrame(res["top"], columns=["Clase", "Probabilidad (%)"])
    resumen = f"Predicción: **{res['label']}** ({res['confidence']:.2f}%)"
    return resumen, table

def build_and_launch_gradio():
    with gr.Blocks(title="FireDoor Assistant - LLaMA3 8-bit offload + Clasificador Híbrido") as demo:
        gr.Markdown("## FireDoor Assistant\nChat Meta LLaMA-3 8B (8-bit con offload CPU) + Clasificador (BERT+RoBERTa)\n\nComando útil: **/clasificar** _texto_")

        with gr.Tab("💬 Chat"):
            chatbot = gr.Chatbot(height=500, type="messages")
            txt = gr.Textbox(show_label=False, placeholder="Escribe aquí... (tip: /clasificar <texto> para clasificar)", lines=2)
            clear = gr.Button("Limpiar historial")
            txt.submit(gradio_chat_fn, [txt, chatbot], [chatbot, txt])
            clear.click(lambda: [], None, chatbot)

        with gr.Tab("🧪 Clasificador"):
            in_txt = gr.Textbox(label="Descripción del defecto", lines=4, placeholder="Escribe la descripción a clasificar…")
            btn = gr.Button("Clasificar")
            out_md = gr.Markdown()
            out_tbl = gr.Dataframe(headers=["Clase", "Probabilidad (%)"], interactive=False)
            btn.click(gradio_classify_fn, [in_txt], [out_md, out_tbl])

        try:
            dev = next(llama_model.parameters()).device
            dtype = next(llama_model.parameters()).dtype
            device_info = f"**Device:** {dev} &nbsp;&nbsp; **dtype:** {dtype} &nbsp;&nbsp; **CUDA:** {torch.cuda.is_available()}"
        except Exception:
            device_info = f"**Device:** N/A &nbsp;&nbsp; **CUDA:** {torch.cuda.is_available()}"
        gr.Markdown(device_info)

    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)

# ===========
# MAIN
# ===========
if __name__ == "__main__":
    try:
        build_and_launch_gradio()
    except Exception as e:
        print("Error lanzando Gradio:", e)
        traceback.print_exc()
        sys.exit(1)
