#!/usr/bin/env python3
# Chat guiado con plantillas + clasificación incremental en "Observaciones"
# LLaMA-3 8B (4-bit si hay VRAM; fallback 8-bit offload) + Clasificador BERT+RoBERTa (CPU)
# Gradio chat (type="messages") con /plantillas, /nuevo <plantilla>, /descargar, /cancelar

import os, sys, traceback
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import gradio as gr
from datetime import datetime

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification
)
from huggingface_hub import login as hf_login, HfFolder
import joblib

# ---------- DOCX opcional ----------
try:
    from docx import Document  # py -m pip install python-docx
except Exception:
    Document = None

# =============== Config LLM ===============
MODEL_NAME         = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_INPUT_TOKENS   = 768
MAX_NEW_TOKENS     = 256
TEMPERATURE        = 0.7
TOP_P              = 0.9
TOP_K              = 40
REPETITION_PENALTY = 1.1

# =============== Token HF ===============
HF_TOKEN = (os.environ.get("REMOVED_TOKEN") or "REMOVED_TOKEN").strip()
if HF_TOKEN and HF_TOKEN != "TU_TOKEN_HF":
    try:
        hf_login(HF_TOKEN)
        HfFolder.save_token(HF_TOKEN)
        print("Hugging Face: autenticado.")
    except Exception as e:
        print("Advertencia HF:", e)
else:
    print("⚠️ No se detectó HF_TOKEN válido; si el repo es gated, fallará al descargar.")

# =============== Utilidades VRAM/DTYPE ===============
def _compute_dtype():
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32

def _gpu_mem_ok(min_free_gib=6.0):
    if not torch.cuda.is_available():
        return False, "CUDA no disponible"
    try:
        free, total = torch.cuda.mem_get_info()
        free_gib = free / (1024**3)
        return (free_gib >= min_free_gib), f"VRAM libre: {free_gib:.1f} GiB"
    except Exception as e:
        return True, f"No se pudo consultar VRAM: {e}"

def _truncate_input_ids(tokenizer, input_ids, max_len):
    if input_ids.shape[1] <= max_len: return input_ids
    return input_ids[:, -max_len:]

# =============== Tokenizer ===============
print("Cargando tokenizer ...")
try:
    llama_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN, use_fast=True)
    if llama_tokenizer.pad_token_id is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token
except Exception as e:
    print("Error cargando tokenizer:", e); sys.exit(1)

# =============== Cargar LLM (4-bit → fallback 8-bit offload) ===============
llama_model = None
if torch.cuda.is_available():
    ok, msg = _gpu_mem_ok(6.0)
    print(msg)
    # Intento 4-bit total-GPU
    try:
        print("Intentando 4-bit NF4 (todo en GPU) ...")
        bnb_4bit = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=_compute_dtype(),
        )
        llama_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_4bit,
            device_map={"": 0},  # todo en cuda:0
            trust_remote_code=False,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            token=HF_TOKEN,
        )
        llama_model.eval()
        print("✅ LLaMA-3 8B cargado en 4-bit (NF4) 100% GPU.")
    except Exception as e4:
        print("❌ 4-bit no entró. Fallback a 8-bit con offload CPU ...")
        try:
            from transformers import BitsAndBytesConfig
            bnb_8bit = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
            offload_dir = "./offload_llama3_8bit"
            os.makedirs(offload_dir, exist_ok=True)
            llama_model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=bnb_8bit,
                device_map="auto",
                offload_folder=offload_dir,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                token=HF_TOKEN,
            )
            llama_model.eval()
            print("✅ LLaMA-3 8B cargado en 8-bit con offload GPU+CPU.")
        except Exception:
            print("❌ Error en fallback 8-bit:")
            traceback.print_exc(); sys.exit(1)
else:
    print("❌ No hay CUDA. Este flujo requiere GPU."); sys.exit(1)

# =============== Clasificador (CPU) ===============
DEVICE_CLF = torch.device("cpu")
BERT_CKPT     = "bert-finetuned-epoch9-acc0.9783"
ROBERTA_CKPT  = "roberta-finetuned-epoch8-acc0.9706"
LABEL_ENCODER = "label_encoder.pkl"
MAX_LEN_CLF   = 64
BEST_W        = 0.50

CLASS_NAMES = []
bert = roberta = bert_tok = rob_tok = None

def _try_load_classifier():
    global bert, roberta, bert_tok, rob_tok, CLASS_NAMES
    try:
        bert_tok = BertTokenizer.from_pretrained(BERT_CKPT, local_files_only=True)
        bert = BertForSequenceClassification.from_pretrained(BERT_CKPT, local_files_only=True).to(DEVICE_CLF)
        rob_tok = RobertaTokenizer.from_pretrained(ROBERTA_CKPT, local_files_only=True)
        roberta = RobertaForSequenceClassification.from_pretrained(ROBERTA_CKPT, local_files_only=True).to(DEVICE_CLF)
        label_encoder = joblib.load(LABEL_ENCODER)
        CLASS_NAMES = list(label_encoder.classes_)
        print("Clasificador híbrido cargado (local, CPU).")
        return True
    except Exception as e_local:
        print("Local falló:", e_local)
        try:
            bert_tok = BertTokenizer.from_pretrained(BERT_CKPT, token=HF_TOKEN)
            bert = BertForSequenceClassification.from_pretrained(BERT_CKPT, token=HF_TOKEN).to(DEVICE_CLF)
            rob_tok = RobertaTokenizer.from_pretrained(ROBERTA_CKPT, token=HF_TOKEN)
            roberta = RobertaForSequenceClassification.from_pretrained(ROBERTA_CKPT, token=HF_TOKEN).to(DEVICE_CLF)
            label_encoder = joblib.load(LABEL_ENCODER)
            CLASS_NAMES = list(label_encoder.classes_)
            print("Clasificador híbrido cargado (descargado, CPU).")
            return True
        except Exception as e_remote:
            print("Error cargando clasificador:", e_remote)
            return False

_classifier_ok = _try_load_classifier()

@torch.no_grad()
def classify_text_hybrid(text: str, best_w: float = BEST_W, topk: int = 3):
    if not _classifier_ok or not (bert and roberta and bert_tok and rob_tok and CLASS_NAMES):
        return {"error": "Clasificador no cargado. Revisa checkpoints y label_encoder.pkl."}
    eb = bert_tok(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN_CLF).to(DEVICE_CLF)
    er =  rob_tok(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN_CLF).to(DEVICE_CLF)
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

# =============== Plantillas ===============
TEMPLATES = {
    "informe_incidente": {
        "title": "Informe de Incidente",
        "fields": [
            {"key": "fecha", "label": "Fecha del incidente", "required": True, "hint":"Ej: 08/10/2025"},
            {"key": "lugar", "label": "Lugar", "required": True, "hint":"Ej: Planta A - Línea 3"},
            {"key": "responsable", "label": "Responsable", "required": True, "hint":"Nombre y cargo"},
            {"key": "descripcion", "label": "Descripción", "required": True, "hint":"¿Qué ocurrió? Detalles clave"},
            {"key": "causa", "label": "Causa probable", "required": False, "hint":"Posible causa raíz"},
            {"key": "acciones", "label": "Acciones correctivas", "required": False, "hint":"Medidas aplicadas y plan"},
            {"key": "observaciones", "label": "Observaciones", "required": False, "hint":"Notas a clasificar"},
        ],
    },
    "carta_solicitud": {
        "title": "Carta de Solicitud",
        "fields": [
            {"key": "fecha", "label": "Fecha", "required": True},
            {"key": "destinatario", "label": "Destinatario", "required": True},
            {"key": "remitente", "label": "Remitente", "required": True},
            {"key": "asunto", "label": "Asunto", "required": True},
            {"key": "cuerpo", "label": "Cuerpo", "required": True},
            {"key": "despedida", "label": "Despedida", "required": False},
            {"key": "firma", "label": "Firma", "required": False},
        ],
    },
}

# =============== Wizard por chat (estado) ===============
FOLLOWUP_QUESTIONS = [
    ("severidad", "En una escala del **1 al 5**, ¿qué tan severo es el defecto (1=leve, 5=crítico)?"),
    ("tamano", "¿Cuál es el **tamaño** aproximado en **cm** (largo x ancho x profundo si aplica)?"),
    ("ubicacion", "¿Dónde se ubica exactamente el defecto? (ej. borde, centro, bisagra, superficie exterior/interior)"),
    ("material", "¿En qué **material/superficie** aparece? (acero, aluminio, pintura, vidrio, plástico, etc.)"),
    ("patron", "¿Qué **patrón** observas? (rayas lineales, abolladura puntual, corrosión/discoloración, grieta, rebaba, etc.)"),
    ("causa", "Si tuvieras que estimar, ¿cuál sería la **causa probable**? (impacto, abrasión, corrosión, fabricación, manejo, transporte)"),
]

CONF_OK = 75.0    # confianza mínima aceptable
MARGIN_OK = 10.0  # diferencia mínima top1-top2

def _reset_session():
    return {
        "active": False,
        "template_key": None,
        "fields": [],
        "idx": 0,
        "answers": {},
        "awaiting_followup": False,
        "followup_queue": [],
        "followup_answers": {},
        "rounds": 0,
        "cls_result": None,
        "file_path": None,
        "finished": False,
    }

def _render_next_question(state):
    f = state["fields"][state["idx"]]
    req = " (obligatorio)" if f.get("required") else ""
    hint = f.get("hint") or ""
    return f"**{f['label']}{req}:**\n_{hint}_"

def _reclassify_with_context(obs_text, extra_dict):
    # Enriquecemos el texto con followups para re-clasificar
    enrich = []
    for k, v in extra_dict.items():
        if v:
            enrich.append(f"{k}: {v}")
    combined = obs_text.strip()
    if enrich:
        combined += "\n" + "\n".join(f"- {e}" for e in enrich)
    return classify_text_hybrid(combined), combined

def _need_more_questions(res):
    if "error" in res: return False
    if res["confidence"] >= CONF_OK:
        # También chequea margen con top-2 si existe
        if res.get("top") and len(res["top"]) >= 2:
            diff = res["top"][0][1] - res["top"][1][1]
            return diff < MARGIN_OK
        return False
    return True

def _enqueue_followups():
    # una cola fija en este ejemplo; podrías personalizar por clases
    return FOLLOWUP_QUESTIONS.copy()

def _generate_doc(template_key, answers, cls_res):
    title = TEMPLATES[template_key]["title"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.getcwd(), "outputs"); os.makedirs(out_dir, exist_ok=True)

    if Document is not None:
        doc = Document()
        doc.add_heading(title, level=1)
        for f in TEMPLATES[template_key]["fields"]:
            k = f["key"]; v = str(answers.get(k, "")).strip()
            p = doc.add_paragraph()
            p.add_run(f"{f['label']}: ").bold = True
            p.add_run(v if v else "—")
        if "observaciones" in answers and answers["observaciones"].strip():
            doc.add_paragraph("")
            doc.add_heading("Clasificación automática de Observaciones", level=2)
            if cls_res and "error" not in cls_res:
                p = doc.add_paragraph()
                p.add_run("Etiqueta predicha: ").bold = True
                p.add_run(f"{cls_res['label']} ({cls_res['confidence']:.2f}%)")
                if cls_res.get("top"):
                    p2 = doc.add_paragraph()
                    p2.add_run("Top: ").bold = True
                    p2.add_run(", ".join([f"{l} ({p:.1f}%)" for l,p in cls_res["top"]]))
            else:
                doc.add_paragraph("Clasificador no disponible.")
        fpath = os.path.join(out_dir, f"{title}_{ts}.docx"); doc.save(fpath)
    else:
        fpath = os.path.join(out_dir, f"{title}_{ts}.txt")
        with open(fpath, "w", encoding="utf-8") as fh:
            fh.write(f"{title}\n\n")
            for f in TEMPLATES[template_key]["fields"]:
                k = f["key"]; v = str(answers.get(k, "")).strip()
                fh.write(f"{f['label']}: {v if v else '—'}\n")
            if "observaciones" in answers and answers["observaciones"].strip():
                fh.write("\n[Clasificación automática de Observaciones]\n")
                if cls_res and "error" not in cls_res:
                    fh.write(f"Etiqueta predicha: {cls_res['label']} ({cls_res['confidence']:.2f}%)\n")
                    if cls_res.get("top"):
                        fh.write("Top: " + ", ".join([f"{l} ({p:.1f}%)" for l,p in cls_res["top"]]) + "\n")
                else:
                    fh.write("Clasificador no disponible.\n")
    return fpath

# =============== Generación LLM (chat) ===============
@torch.no_grad()
def llama_chat_generate(user_message: str, history_pairs: list | None = None) -> str:
    history_pairs = history_pairs or []
    messages = [{"role":"system","content":"You are a helpful assistant."}]
    for u,a in history_pairs:
        messages += [{"role":"user","content":u},{"role":"assistant","content":a}]
    messages.append({"role":"user","content":user_message})

    prompt = llama_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = llama_tokenizer(prompt, return_tensors="pt", truncation=False)
    enc["input_ids"] = _truncate_input_ids(llama_tokenizer, enc["input_ids"], MAX_INPUT_TOKENS)
    if "attention_mask" in enc:
        enc["attention_mask"] = enc["attention_mask"][:, -enc["input_ids"].shape[1]:]

    device = next(llama_model.parameters()).device
    inputs = {k: v.to(device) for k,v in enc.items()}

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
        use_cache=True,
    )
    gen_ids = out[0]; input_len = inputs["input_ids"].shape[1]
    new_tokens = gen_ids[input_len:]
    return llama_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# =============== Chat Handler con Wizard Integrado ===============
def chat_handler(message, chat_history, session_state, file_comp):
    chat_history = chat_history or []
    state = session_state or _reset_session()
    user_msg = (message or "").strip()

    # util para construir historial pares
    pairs, last_user = [], None
    for m in chat_history:
        if m["role"] == "user": last_user = m["content"]
        elif m["role"] == "assistant" and last_user is not None:
            pairs.append((last_user, m["content"])); last_user = None

    def send(bot_text):
        chat_history.append({"role":"user","content":user_msg})
        chat_history.append({"role":"assistant","content":bot_text})
        return chat_history

    # --------- Comandos ---------
    if user_msg.lower() == "/plantillas":
        names = "\n".join([f"- {meta['title']}  (`/nuevo {k}`)" for k,meta in TEMPLATES.items()])
        return send(f"Plantillas disponibles:\n{names}"), "", state, gr.update()

    if user_msg.lower().startswith("/nuevo"):
        parts = user_msg.split()
        if len(parts) < 2 or parts[1] not in TEMPLATES:
            hint = "Usa: /nuevo informe_incidente  ó  /nuevo carta_solicitud"
            return send(f"Plantilla no reconocida.\n{hint}"), "", _reset_session(), gr.update()
        key = parts[1]
        state = _reset_session()
        state["active"] = True
        state["template_key"] = key
        state["fields"] = TEMPLATES[key]["fields"]
        state["idx"] = 0
        q = _render_next_question(state)
        return send(f"Has iniciado **{TEMPLATES[key]['title']}**.\n{q}"), "", state, gr.update(value=None)

    if user_msg.lower() == "/cancelar":
        state = _reset_session()
        return send("Sesión cancelada. Escribe `/plantillas` para ver opciones."), "", state, gr.update()

    if user_msg.lower() == "/descargar":
        if not state["active"] or not state["finished"]:
            return send("Aún no has completado la plantilla. Usa `/nuevo <plantilla>` para iniciar."), "", state, gr.update()
        fpath = _generate_doc(state["template_key"], state["answers"], state["cls_result"])
        state["file_path"] = fpath
        return send(f"Documento generado: **{os.path.basename(fpath)}** (ver debajo para descargar)"), "", state, gr.update(value=fpath)

    # --------- Flujo guiado si hay sesión activa ---------
    if state["active"] and not state["finished"]:
        # Si estamos en preguntas de seguimiento (observaciones)
        if state["awaiting_followup"] and state["followup_queue"]:
            key, _ = state["followup_queue"].pop(0)
            state["followup_answers"][key] = user_msg

            # ¿Más followups?
            if state["followup_queue"]:
                next_q = state["followup_queue"][0][1]
                return send(f"Gracias. {next_q}"), "", state, gr.update()

            # Re-clasificamos con info adicional
            obs = state["answers"].get("observaciones", "")
            res, combined = _reclassify_with_context(obs, state["followup_answers"])
            state["answers"]["observaciones"] = combined  # guardamos texto enriquecido
            state["cls_result"] = res
            state["rounds"] += 1

            if "error" in res:
                state["awaiting_followup"] = False
                state["finished"] = True
                return send(f"Clasificador no disponible: {res['error']}\nHas completado la plantilla. Escribe **/descargar** para generar el documento."), "", state, gr.update()

            msg = f"**Clasificación provisional**: {res['label']} ({res['confidence']:.2f}%)"
            if res.get("top"):
                msg += "\nTop: " + ", ".join([f"{l} ({p:.1f}%)" for l,p in res["top"]])

            if (_need_more_questions(res) and state["rounds"] < 2):
                state["followup_queue"] = _enqueue_followups()
                state["awaiting_followup"] = True
                next_q = state["followup_queue"][0][1]
                return send(f"{msg}\n\nNecesito unos datos más para afinar:\n{next_q}"), "", state, gr.update()
            else:
                state["awaiting_followup"] = False
                state["finished"] = True
                return send(f"{msg}\n\n¡Listo! Has completado la plantilla. Escribe **/descargar** para obtener el archivo."), "", state, gr.update()

        # Recolectamos el campo actual
        f = state["fields"][state["idx"]]
        # Validación simple
        if f.get("required") and not user_msg:
            return send(f"El campo **{f['label']}** es obligatorio. Intenta nuevamente."), "", state, gr.update()
        state["answers"][f["key"]] = user_msg

        # Si acabamos de llenar 'observaciones', disparamos clasificación incremental
        just_obs = (f["key"] == "observaciones")
        state["idx"] += 1

        if just_obs and user_msg.strip():
            res = classify_text_hybrid(user_msg)
            state["cls_result"] = res
            if "error" in res:
                state["finished"] = True
                return send(f"Clasificador no disponible: {res['error']}\nHas completado la plantilla. Escribe **/descargar** para generar el documento."), "", state, gr.update()

            msg = f"**Clasificación provisional**: {res['label']} ({res['confidence']:.2f}%)"
            if res.get("top"):
                msg += "\nTop: " + ", ".join([f"{l} ({p:.1f}%)" for l,p in res["top"]])

            if _need_more_questions(res):
                state["followup_queue"] = _enqueue_followups()
                state["awaiting_followup"] = True
                state["rounds"] = 0
                next_q = state["followup_queue"][0][1]
                return send(f"{msg}\n\nPara ser más precisos, responde:\n{next_q}"), "", state, gr.update()

        # ¿Quedan más campos?
        if state["idx"] < len(state["fields"]):
            q = _render_next_question(state)
            return send(q), "", state, gr.update()

        # No quedan campos → si no hubo observaciones o ya acabamos followups: finalizar
        state["finished"] = True
        return send("¡Plantilla completa! Escribe **/descargar** para generar el documento."), "", state, gr.update()

    # --------- Modo chat normal (sin sesión) ---------
    try:
        reply = llama_chat_generate(user_msg, pairs)
    except Exception as e:
        reply = f"⚠️ Error generando respuesta: {e}"
    chat_history.append({"role":"user","content":user_msg})
    chat_history.append({"role":"assistant","content":reply})
    return chat_history, "", state, gr.update()

# =============== UI Gradio ===============
def build_and_launch_gradio():
    with gr.Blocks(title="Chat Plantillas + Clasificación Incremental") as demo:
        gr.Markdown("## Chat de Plantillas con Clasificación\n"
                    "- `/plantillas` para listar\n"
                    "- `/nuevo informe_incidente` o `/nuevo carta_solicitud` para iniciar\n"
                    "- `/descargar` cuando termines\n"
                    "- `/cancelar` para abortar\n\n"
                    "Al llegar a **Observaciones**, te haré preguntas para **afinar la clasificación**.")

        chatbot = gr.Chatbot(height=520, type="messages")
        txt = gr.Textbox(show_label=False, placeholder="Escribe aquí…", lines=2)
        file_out = gr.File(label="Descargar documento", interactive=False)
        state = gr.State(_reset_session())

        txt.submit(chat_handler, [txt, chatbot, state, file_out], [chatbot, txt, state, file_out])

        try:
            dev = next(llama_model.parameters()).device
            dtype = next(llama_model.parameters()).dtype
            gr.Markdown(f"**LLM Device:** {dev} &nbsp;&nbsp; **dtype:** {dtype} &nbsp;&nbsp; **CUDA:** {torch.cuda.is_available()}")
        except Exception:
            gr.Markdown(f"**CUDA:** {torch.cuda.is_available()}")

    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)

# =============== MAIN ===============
if __name__ == "__main__":
    try:
        build_and_launch_gradio()
    except Exception as e:
        print("Error lanzando Gradio:", e)
        traceback.print_exc(); sys.exit(1)
