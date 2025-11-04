import gradio as gr
import os
import PyPDF2
import logging
import torch
import threading
import time
import re, unicodedata

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers import logging as hf_logging
import spaces
from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Document as LlamaDocument,
)
from llama_index.core import Settings
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_leaf_nodes,
    get_root_nodes,
)
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
hf_logging.set_verbosity_error()

MEDSWIN_MODEL = "MedAI-COS30018/MedSwin-7B-Distilled"
MEDALPACA_MODEL = "medalpaca/medalpaca-7b"
MODEL = MEDSWIN_MODEL
EMBEDDING_MODEL = "abhinand/MedEmbed-large-v0.1"
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment variables")

# Custom UI
TITLE = "<h1><center>Medical RAG Assistant (MedSwin-7B Distilled)</center></h1>"
DESCRIPTION = """
<center>
<p>Upload clinical PDFs or text (guidelines, notes, literature) to build a medical context.</p>
<p>This app retrieves relevant snippets and answers with our specialized medical LLM.</p>
<p><b>Important:</b> This is an information tool, not a substitute for professional medical advice.</p>
</center>
"""
CSS = """
.upload-section {
    max-width: 400px;
    margin: 0 auto;
    padding: 10px;
    border: 2px dashed #ccc;
    border-radius: 10px;
}
.upload-button {
    background: #2e7d32 !important; /* medical green */
    color: white !important;
    border-radius: 25px !important;
}
.chatbot-container {
    margin-top: 20px;
}
.status-output {
    margin-top: 10px;
    font-size: 14px;
}
.processing-info {
    margin-top: 5px;
    font-size: 12px;
    color: #666;
}
.info-container {
    margin-top: 10px;
    padding: 10px;
    border-radius: 5px;
}
.file-list {
    margin-top: 0;
    max-height: 200px;
    overflow-y: auto;
    padding: 5px;
    border: 1px solid #eee;
    border-radius: 5px;
}
.stats-box {
    margin-top: 10px;
    padding: 10px;
    border-radius: 5px;
    font-size: 12px;
}
.submit-btn {
    background: #00796b !important; /* teal */
    color: white !important;
    border-radius: 25px !important;
    margin-left: 10px;
    padding: 5px 10px;
    font-size: 16px;
}
.input-row {
    display: flex;
    align-items: center;
}
@media (min-width: 768px) {
    .main-container {
        display: flex;
        justify-content: space-between;
        gap: 20px;
    }
    .upload-section {
        flex: 1;
        max-width: 300px;
    }
    .chatbot-container {
        flex: 2;
        margin-top: 0;
    }
}
"""

global_model = None
global_tokenizer = None
global_file_info = {}
model_cache = {}


import html

def _normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = html.unescape(s)                     # &nbsp; &amp; → real chars
    s = s.replace("\u00A0", " ")             # NBSP → space
    s = re.sub(r"[\[\]\{\}\(\)<>/*_+=\-]{3,}", " ", s)

    keep = []
    for line in s.splitlines():
        letters = sum(ch.isalnum() for ch in line)
        punct   = sum(ch in r"[]{}()<>/*_+=\-|~`^" for ch in line)
        if letters == 0 and punct > 0:           # junk lines
            continue
        if letters and punct / max(1, letters) > 1.5:
            continue
        keep.append(line)
    s = "\n".join(keep)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _truncate_by_tokens(text: str, tokenizer, max_tokens: int = 1800) -> str:
    ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[-max_tokens:], skip_special_tokens=True)


DISCLAIMER_PATTERNS = [
    r"\bnot (a|your) (doctor|physician)\b",
    r"\bnot medical advice\b",
    r"\bfor informational purposes only\b",
    r"\balways consult (a|your) healthcare (professional|provider)\b",
    r"\bas an ai\b",
    r"\bi (cannot|can't) provide medical advice\b",
]

def _strip_disclaimers(text: str) -> str:
    t = text
    for pat in DISCLAIMER_PATTERNS:
        t = re.sub(pat, "", t, flags=re.I)
    # collapse leftover double spaces
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def _build_fallback_chat_prompt(messages):
    # Alpaca-style fallback prompt that works well with MedAlpaca/Gemma-derived SFTs
    # We collapse system + last user turn into an Instruction, keep brief history inline
    sys_blocks = [m.get("content", "").strip() for m in messages if m.get("role") == "system"]
    sys_text = "\n".join([b for b in sys_blocks if b])
    user_turns = [m.get("content", "").strip() for m in messages if m.get("role") == "user"]
    last_user = user_turns[-1] if user_turns else ""

    history_pairs = []
    current_q = None
    for m in messages:
        role = m.get("role")
        content = (m.get("content", "") or "").strip()
        if role == "user":
            current_q = content
        elif role == "assistant" and current_q:
            history_pairs.append((current_q, content))
            current_q = None

    history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in history_pairs[-2:]])  # keep last 2 QA pairs

    instruction = sys_text
    if history_text:
        instruction += f"\n\nContext Conversation (for background):\n{history_text}"
    if last_user:
        instruction += f"\n\nTask: Answer the user's question.\nQuestion: {last_user}"

    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
    )


def build_prompt(messages, tokenizer, system_prompt: str, context: str, source_info: str):
    sys = system_prompt.strip()
    if context:
        sys = f"{sys}\n\n[Document Context]\n{context}{source_info}"

    # Prefer chat template when available (chat-tuned models)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            chat_msgs = [{"role": "system", "content": sys}]
            for m in messages:
                # Gradio `type="messages"` gives {"role": "...", "content": "..."}
                if m.get("role") in ("user", "assistant", "system"):
                    chat_msgs.append({"role": m["role"], "content": m.get("content","")})
            return tokenizer.apply_chat_template(
                chat_msgs, tokenize=False, add_generation_prompt=True
            ), True
        except Exception:
            pass

    # Fallback: instruct-style (works for non-chat causal LMs)
    # Strong guardrails to avoid “As an AI…” style preambles
    instruct = (
        "### System\n"
        f"{sys}\n\n"
        "### Rules\n"
        "- Answer directly in clinical language.\n"
        "- Do not mention that you are an AI.\n"
        "- Do not include meta commentary or apologies.\n"
        "- Keep it concise and evidence-focused.\n\n"
        "### Conversation\n"
    )
    for m in messages:
        role = m.get("role","user")
        content = (m.get("content","") or "").strip()
        if role == "user":
            instruct += f"User: {content}\n"
        elif role == "assistant":
            instruct += f"Assistant: {content}\n"
    instruct += "Assistant: "
    return instruct, False


def _select_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.float32
    return torch.float32

def _load_model_and_tokenizer(model_name: str):
    dtype = _select_dtype()
    logger.info(f"Loading model={model_name} dtype={dtype}")
    try:
        tok = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    except ValueError as e:
        logger.warning(f"Fast tokenizer load failed ({e}). Retrying with slow tokenizer...")
        tok = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN, use_fast=False)

    if tok.eos_token_id is None and getattr(tok, "eos_token", None) is None:
        tok.eos_token = "</s>"
    if tok.pad_token_id is None and getattr(tok, "pad_token", None) is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
    tok.padding_side = "right"

    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    if tok.pad_token_id is not None:
        mdl.config.pad_token_id = tok.pad_token_id
        mdl.generation_config.pad_token_id = tok.pad_token_id
    if tok.eos_token_id is not None:
        mdl.config.eos_token_id = tok.eos_token_id
        mdl.generation_config.eos_token_id = tok.eos_token_id

    if hasattr(mdl, "resize_token_embeddings"):
        try:
            mdl.resize_token_embeddings(len(tok))
        except Exception:
            pass
    return tok, mdl

def initialize_model_and_tokenizer(model_name: str = None):
    global global_model, global_tokenizer, MODEL
    name = model_name or MODEL
    if name in model_cache:
        global_tokenizer, global_model = model_cache[name]
        MODEL = name
        return
    tok, mdl = _load_model_and_tokenizer(name)
    model_cache[name] = (tok, mdl)
    global_tokenizer, global_model = tok, mdl
    MODEL = name


def get_llm(temperature=0.0, max_new_tokens=256, top_p=0.95, top_k=50, model_name: str = None):
    global global_model, global_tokenizer
    if global_model is None or global_tokenizer is None:
        initialize_model_and_tokenizer(model_name)
    elif model_name:
        initialize_model_and_tokenizer(model_name)
    
    return HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=max_new_tokens,
        tokenizer=global_tokenizer,
        model=global_model,
        generate_kwargs={
            "do_sample": False,
            "temperature": temperature,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 4,
            "top_k": top_k,
            "top_p": top_p
        }
    )


def extract_text_from_document(file):
    file_name = file.name
    file_extension = os.path.splitext(file_name)[1].lower()
    
    if file_extension == '.txt':
        text = file.read().decode('utf-8')
        return text, len(text.split()), None
    elif file_extension == '.pdf':
        pdf_reader = PyPDF2.PdfReader(file)
        text = "\n\n".join(page.extract_text() for page in pdf_reader.pages)
        return text, len(text.split()), None
    else:
        return None, 0, ValueError(f"Unsupported file format: {file_extension}")


@spaces.GPU()
def create_or_update_index(files, request: gr.Request):
    global global_file_info
    
    if not files:
        return "Please provide files.", ""
    
    start_time = time.time()
    user_id = request.session_hash
    save_dir = f"./{user_id}_index"
    # Initialize LlamaIndex modules
    llm = get_llm(model_name=MODEL)
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL, token=HF_TOKEN)
    Settings.llm = llm
    Settings.embed_model = embed_model
    file_stats = []
    new_documents = []
    
    for file in tqdm(files, desc="Processing files"):
        file_basename = os.path.basename(file.name)
        text, word_count, error = extract_text_from_document(file)
        if error:
            logger.error(f"Error processing file {file_basename}: {str(error)}")
            file_stats.append({
                "name": file_basename,
                "words": 0,
                "status": f"error: {str(error)}"
            })
            continue
        
        doc = LlamaDocument(
            text=text,
            metadata={
                "file_name": file_basename,
                "word_count": word_count,
                "source": "user_upload"
            }
        )
        new_documents.append(doc)
        
        file_stats.append({
            "name": file_basename,
            "words": word_count,
            "status": "processed"
        })
        
        global_file_info[file_basename] = {
            "word_count": word_count,
            "processed_at": time.time()
        }
    
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[2048, 512, 128],  
        chunk_overlap=20         
    )
    logger.info(f"Parsing {len(new_documents)} documents into hierarchical nodes")
    new_nodes = node_parser.get_nodes_from_documents(new_documents)
    new_leaf_nodes = get_leaf_nodes(new_nodes)
    new_root_nodes = get_root_nodes(new_nodes)
    logger.info(f"Generated {len(new_nodes)} total nodes ({len(new_root_nodes)} root, {len(new_leaf_nodes)} leaf)")
    node_ancestry = {}
    for node in new_nodes:
        if hasattr(node, 'metadata') and 'file_name' in node.metadata:
            file_origin = node.metadata['file_name']
            if file_origin not in node_ancestry:
                node_ancestry[file_origin] = 0
            node_ancestry[file_origin] += 1
    
    if os.path.exists(save_dir):
        logger.info(f"Loading existing index from {save_dir}")
        storage_context = StorageContext.from_defaults(persist_dir=save_dir)
        index = load_index_from_storage(storage_context, settings=Settings)
        docstore = storage_context.docstore
        
        docstore.add_documents(new_nodes)
        for node in tqdm(new_leaf_nodes, desc="Adding leaf nodes to index"):
            index.insert_nodes([node])
            
        total_docs = len(docstore.docs)
        logger.info(f"Updated index with {len(new_nodes)} new nodes from {len(new_documents)} files")
    else:
        logger.info("Creating new index")
        docstore = SimpleDocumentStore()
        storage_context = StorageContext.from_defaults(docstore=docstore)
        docstore.add_documents(new_nodes)
        
        index = VectorStoreIndex(
            new_leaf_nodes, 
            storage_context=storage_context, 
            settings=Settings
        )
        total_docs = len(new_documents)
        logger.info(f"Created new index with {len(new_nodes)} nodes from {len(new_documents)} files")
    
    index.storage_context.persist(persist_dir=save_dir)
    # custom outputs after processing files
    file_list_html = "<div class='file-list'>"
    for stat in file_stats:
        status_color = "#4CAF50" if stat["status"] == "processed" else "#f44336"
        file_list_html += f"<div><span style='color:{status_color}'>●</span> {stat['name']} - {stat['words']} words</div>"
    file_list_html += "</div>"
    processing_time = time.time() - start_time
    stats_output = f"<div class='stats-box'>"
    stats_output += f"✓ Processed {len(files)} files in {processing_time:.2f} seconds<br>"
    stats_output += f"✓ Created {len(new_nodes)} nodes ({len(new_leaf_nodes)} leaf nodes)<br>"
    stats_output += f"✓ Total documents in index: {total_docs}<br>"
    stats_output += f"✓ Index saved to: {save_dir}<br>"
    stats_output += "</div>"
    output_container = f"<div class='info-container'>"
    output_container += file_list_html
    output_container += stats_output
    output_container += "</div>"
    return f"Successfully indexed {len(files)} files.", output_container


@spaces.GPU()
def stream_chat(
    message: str,
    history: list,
    system_prompt: str,
    disable_retrieval: bool,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    penalty: float,
    retriever_k: int,
    merge_threshold: float,
    request: gr.Request
):
    # --- guards & basics ---
    if not request:
        yield history + [{"role": "assistant", "content": "Session initialization failed. Please refresh the page."}]
        return
    user_id = request.session_hash
    index_dir = f"./{user_id}_index"

    # normalize UI params
    max_new_tokens = int(max_new_tokens) if isinstance(max_new_tokens, (int, float)) else 1024
    temperature    = float(temperature)  if isinstance(temperature,  (int, float)) else 0.0
    top_p          = float(top_p)        if isinstance(top_p,        (int, float)) else 0.95
    top_k          = int(top_k)          if isinstance(top_k,        (int, float)) else 50
    penalty        = float(penalty)      if isinstance(penalty,      (int, float)) else 1.2
    retriever_k    = int(retriever_k)    if isinstance(retriever_k,  (int, float)) else 15
    merge_threshold= float(merge_threshold) if isinstance(merge_threshold, (int, float)) else 0.5

    # ensure model/tokenizer exist (uses currently selected MODEL)
    if global_model is None or global_tokenizer is None:
        initialize_model_and_tokenizer(MODEL)

    # --- retrieval (optional) ---
    context = ""
    source_info = ""
    try:
        if not disable_retrieval:
            if not os.path.exists(index_dir):
                yield history + [{"role": "assistant", "content": "Please upload documents first or enable 'Disable document retrieval' to chat without documents."}]
                return

            embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL, token=HF_TOKEN)
            Settings.embed_model = embed_model

            storage_context = StorageContext.from_defaults(persist_dir=index_dir)
            index = load_index_from_storage(storage_context, settings=Settings)

            base_retriever = index.as_retriever(similarity_top_k=retriever_k)
            auto_merging_retriever = AutoMergingRetriever(
                base_retriever,
                storage_context=storage_context,
                simple_ratio_thresh=merge_threshold,
                verbose=True
            )

            logger.info(f"[query] {message}")
            t0 = time.time()
            base_nodes = base_retriever.retrieve(message)
            logger.info(f"[retrieval] base={len(base_nodes)} in {time.time()-t0:.2f}s")

            t1 = time.time()
            merged_nodes = auto_merging_retriever.retrieve(message)
            logger.info(f"[retrieval] merged={len(merged_nodes)} in {time.time()-t1:.2f}s")

            # merge text + normalize + truncate by tokens to keep headroom for generation
            context = "\n\n".join([(n.node.text or "") for n in merged_nodes])
            context = _normalize_text(context)
            context = _truncate_by_tokens(context, global_tokenizer, max_tokens=1800)

            # compact source list
            srcs = []
            for n in merged_nodes:
                md = getattr(n.node, "metadata", {}) or {}
                fn = md.get("file_name")
                if fn and fn not in srcs:
                    srcs.append(fn)
            if srcs:
                source_info = "\n\n[Sources] " + ", ".join(srcs)
    except Exception as e:
        logger.exception(f"retrieval error: {e}")
        # fallback to no context rather than failing the chat
        context, source_info = "", ""

    # --- prompt building (template-aware) ---
    sys_text = (system_prompt or "").strip()
    if context:
        sys_text = f"{sys_text}\n\n[Document Context]\n{context}{source_info}"

    # Reconstruct conversation for template
    convo_msgs = [{"role": "system", "content": sys_text}]
    for m in (history or []):
        if m and isinstance(m, dict) and m.get("role") in ("user", "assistant", "system"):
            convo_msgs.append({"role": m["role"], "content": m.get("content", "")})
    convo_msgs.append({"role": "user", "content": message})

    used_chat_template = False
    if hasattr(global_tokenizer, "apply_chat_template"):
        try:
            prompt = global_tokenizer.apply_chat_template(convo_msgs, tokenize=False, add_generation_prompt=True)
            used_chat_template = True
        except Exception:
            # fallback to a clean instruct format
            prompt = _build_fallback_chat_prompt(convo_msgs)
    else:
        prompt = _build_fallback_chat_prompt(convo_msgs)

    # --- streaming infra ---
    stop_event = threading.Event()

    class StopOnEvent(StoppingCriteria):
        def __init__(self, stop_event):
            super().__init__()
            self.stop_event = stop_event
        def __call__(self, input_ids, scores, **kwargs):
            return self.stop_event.is_set()

    stopping_criteria = StoppingCriteriaList([StopOnEvent(stop_event)])
    streamer = TextIteratorStreamer(global_tokenizer, skip_prompt=True, skip_special_tokens=True)

    # fit prompt within context window
    ctx = int(getattr(global_model.config, "max_position_embeddings", 4096))
    max_inp = max(256, ctx - int(max_new_tokens) - 8)
    enc = global_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_inp).to(global_model.device)

    # avoid aggressive bad-words filtering which can distort outputs
    bad_words_ids = None

    # Honour temperature: sample iff temperature > 0
    use_sampling = float(temperature) > 0.0
    min_tokens = 16

    # deterministic clinical generation
    min_tokens = 16
    
    generation_kwargs = dict(
        **enc,
        streamer=streamer,
        max_new_tokens=int(max_new_tokens),
        min_new_tokens=min_tokens,
        do_sample=use_sampling,
        repetition_penalty=max(1.1, float(penalty)),
        no_repeat_ngram_size=4,
        use_cache=True,
        stopping_criteria=stopping_criteria,
        bad_words_ids=bad_words_ids,
    )
    if use_sampling:
        generation_kwargs.update(
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
        )

    logger.info(f"chat_template={'yes' if used_chat_template else 'no'}  ctx={ctx}  max_inp={max_inp}  max_new={max_new_tokens}")
    logger.info(f"prompt_preview={(prompt[:200].replace(chr(10),' '))}")
    thread = threading.Thread(target=global_model.generate, kwargs=generation_kwargs)
    thread.start()

    # prime UI
    updated_history = (history or []) + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
    yield updated_history

    partial_response = ""
    first_token_received = threading.Event()

    def _watch_first_token():
        if not first_token_received.wait(timeout=45):
            logger.warning("Generation timeout: no tokens in 45s; stopping stream.")
            stop_event.set()

    watchdog = threading.Thread(target=_watch_first_token, daemon=True)
    watchdog.start()

    try:
        for chunk in streamer:
            if chunk and not first_token_received.is_set():
                first_token_received.set()
            partial_response += chunk
            updated_history[-1]["content"] = partial_response
            yield updated_history

        # Final tidy-up before returning
        final_text = _strip_disclaimers(_normalize_text(partial_response))
        updated_history[-1]["content"] = final_text
        yield updated_history

    except GeneratorExit:
        stop_event.set()
        thread.join()
        raise
    except Exception as e:
        logger.exception(f"streaming error: {e}")
        stop_event.set()
        updated_history[-1]["content"] = (partial_response or "") + "\n\n[Generation stopped due to an internal error.]"
        yield updated_history


    # remove duplicated second streaming loop


def create_demo():
    with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
        gr.HTML(TITLE)
        gr.HTML(DESCRIPTION)
        
        with gr.Row(elem_classes="main-container"):
            with gr.Column(elem_classes="upload-section"):
                file_upload = gr.File(
                    file_count="multiple",
                    label="Drag and Drop Files Here",
                    file_types=[".pdf", ".txt"],
                    elem_id="file-upload"
                )
                upload_button = gr.Button("Upload & Index", elem_classes="upload-button")
                status_output = gr.Textbox(
                    label="Status",
                    placeholder="Upload files to start...",
                    interactive=False
                )
                file_info_output = gr.HTML(
                    label="File Information",
                    elem_classes="processing-info"
                )
                upload_button.click(
                    fn=create_or_update_index,
                    inputs=[file_upload],
                    outputs=[status_output, file_info_output]
                )
            
            with gr.Column(elem_classes="chatbot-container"):
                chatbot = gr.Chatbot(
                    height=500,
                    placeholder="Chat with your documents here... Type your question below.",
                    show_label=False,
                    type="messages"
                )
                model_selector = gr.Radio(
                    choices=["MedSwin-7B Distilled", "MedAlpaca-7B"],
                    value="MedSwin-7B Distilled",
                    label="Model"
                )
                disable_retrieval = gr.Checkbox(
                    label="Disable document retrieval (use model ground knowledge)",
                    value=False
                )
                with gr.Row(elem_classes="input-row"):
                    message_input = gr.Textbox(
                        placeholder="Type your question here...",
                        show_label=False,
                        container=False,
                        lines=1,
                        scale=8
                    )
                    submit_button = gr.Button("➤", elem_classes="submit-btn", scale=1)
                
                with gr.Accordion("Advanced Settings", open=False):
                    system_prompt = gr.Textbox(
                        value=(
                            "You are a cautious, evidence-focused medical assistant.\n"
                            "- Write directly to the user in second person (use 'you'); never say 'the user' or 'the patient'.\n"
                            "- Be concise (≤150 words). Use clinical language and bullet points only if helpful.\n"
                            "- If context is insufficient, say exactly what is missing. Do not fabricate.\n"
                            "- If you used provided documents, cite filenames in brackets (e.g., [guideline.pdf]).\n"
                            "- Do not include disclaimers, meta-commentary, or references to being an AI."
                        ),
                        label="System Prompt",
                        lines=5
                    )
                    gr.Markdown(
                        "**Clinical Use Disclaimer:** This application is intended for informational purposes only and does not constitute medical advice. "
                        "Always consult qualified healthcare professionals for diagnosis and treatment decisions.")
                    
                    with gr.Tab("Generation Parameters"):
                        temperature = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=0.5,  
                            label="Temperature"
                        )
                        max_new_tokens = gr.Slider(
                            minimum=128,
                            maximum=8192,
                            step=64,
                            value=1024,
                            label="Max New Tokens",
                        )
                        top_p = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.9, 
                            label="Top P"
                        )
                        top_k = gr.Slider(
                            minimum=1,
                            maximum=100,  
                            step=1,
                            value=50,  
                            label="Top K"
                        )
                        penalty = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            step=0.1,
                            value=1.15,
                            label="Repetition Penalty"
                        )
                        
                    with gr.Tab("Retrieval Parameters"):
                        retriever_k = gr.Slider(
                            minimum=5,
                            maximum=30,
                            step=1,
                            value=15,
                            label="Initial Retrieval Size (Top K)"
                        )
                        merge_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            step=0.1,
                            value=0.5,
                            label="Merge Threshold (lower = more merging)"
                        )

                def _on_model_change(choice):
                    name = MEDSWIN_MODEL if choice == "MedSwin-7B Distilled" else MEDALPACA_MODEL
                    initialize_model_and_tokenizer(name)
                    return f"Loaded: {choice}"

                submit_button.click(
                    fn=stream_chat,
                    inputs=[
                        message_input, 
                        chatbot, 
                        system_prompt, 
                        disable_retrieval,
                        temperature, 
                        max_new_tokens, 
                        top_p, 
                        top_k, 
                        penalty,
                        retriever_k,
                        merge_threshold
                    ],
                    outputs=chatbot
                )
                model_selector.change(
                    fn=_on_model_change,
                    inputs=[model_selector],
                    outputs=[]
                )
                submit_button.click(
                    fn=stream_chat,
                    inputs=[
                        message_input, 
                        chatbot, 
                        system_prompt, 
                        disable_retrieval,
                        temperature, 
                        max_new_tokens, 
                        top_p, 
                        top_k, 
                        penalty,
                        retriever_k,
                        merge_threshold
                    ],
                    outputs=chatbot
                )
                
                message_input.submit(
                    fn=stream_chat,
                    inputs=[
                        message_input, 
                        chatbot, 
                        system_prompt, 
                        disable_retrieval,
                        temperature, 
                        max_new_tokens, 
                        top_p, 
                        top_k, 
                        penalty,
                        retriever_k,
                        merge_threshold
                    ],
                    outputs=chatbot
                )
                model_selector.change(
                    fn=_on_model_change,
                    inputs=[model_selector],
                    outputs=[]
                )
                message_input.submit(
                    fn=stream_chat,
                    inputs=[
                        message_input, 
                        chatbot, 
                        system_prompt, 
                        disable_retrieval,
                        temperature, 
                        max_new_tokens, 
                        top_p, 
                        top_k, 
                        penalty,
                        retriever_k,
                        merge_threshold
                    ],
                    outputs=chatbot
                )
    return demo


if __name__ == "__main__":
    initialize_model_and_tokenizer()
    demo = create_demo()
    demo.launch()