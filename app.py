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
    LogitsProcessor,
    LogitsProcessorList,
)
from transformers import logging as hf_logging
from huggingface_hub import login as hf_login
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

# Log if .env was loaded (after logger is initialized)
try:
    from dotenv import load_dotenv
    if load_dotenv():
        logger.info("Loaded environment variables from .env file")
except ImportError:
    pass

MEDSWIN_KD_MODEL = "MedAI-COS30018/MedSwin-7B-KD"
MEDSWIN_SFT_MODEL = "MedAI-COS30018/MedSwin-7B-SFT"
MEDALPACA_MODEL = "medalpaca/medalpaca-7b"
MEDGEMMA_MODEL = "google/medgemma-27b-text-it"
MODEL = MEDSWIN_KD_MODEL
EMBEDDING_MODEL = "abhinand/MedEmbed-large-v0.1"

# Load HF_TOKEN from environment (supports both HF_TOKEN and HUGGINGFACE_HUB_TOKEN)
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN or HUGGINGFACE_HUB_TOKEN not found in environment variables")

# Set token in environment for transformers to pick up automatically (like hf auth login)
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

# Authenticate with HuggingFace Hub (equivalent to hf auth login)
try:
    hf_login(token=HF_TOKEN, add_to_git_credential=False)
    logger.info("Successfully authenticated with HuggingFace Hub")
except Exception as e:
    logger.warning(f"Could not authenticate with HuggingFace Hub: {e}. Will try with explicit token.")

# Custom UI
TITLE = "<h1><center>Medical RAG Assistant (MedSwin-7B)</center></h1>"
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
    # Preserve special tokens for proper decoding, especially for non-English languages
    return tokenizer.decode(ids[-max_tokens:], skip_special_tokens=False)


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


LEADING_FILLER = [
    r"^as an? [^,.:;]{0,80}[:,]?$",
    r"^note:\s*",
    r"^as a small amount.*$",
    r"^the user (is|has|was) .*?$",
]

def _clean_leading_filler(text: str) -> str:
    lines = [l.strip() for l in (text or "").splitlines()]
    i = 0
    while i < len(lines) and i < 2:  # only trim at most first two lines
        line = lines[0]
        if any(re.match(p, line, flags=re.I) for p in LEADING_FILLER):
            lines.pop(0)
        else:
            break
        i += 1
    cleaned = "\n".join(lines).strip()
    # soften third-person phrasing at the start
    cleaned = re.sub(r"\b[Tt]he user\b", "you", cleaned, count=2)
    return cleaned


def _detect_language(text: str) -> str:
    """Simple language detection - returns 'vi' for Vietnamese, 'en' for English, or 'other'"""
    if not text:
        return 'en'
    
    # Vietnamese has distinctive characters: ă, â, đ, ê, ô, ơ, ư
    vietnamese_chars = set('ăâđêôơưĂÂĐÊÔƠƯ')
    text_chars = set(text)
    
    # Count Vietnamese-specific characters
    vi_char_count = len(text_chars & vietnamese_chars)
    # If more than 2% of unique characters are Vietnamese-specific, likely Vietnamese
    if vi_char_count > 0 and len(text) > 20:
        # Check for common Vietnamese words/patterns
        vi_patterns = [
            r'\b(của|và|với|cho|được|trong|này|đó|khi|nếu|vì|nên|nhưng|hoặc)\b',
            r'\b(là|sẽ|có|không|đã|đang|sẽ|bị|bởi)\b',
            r'\b(người|bệnh|thuốc|điều trị|triệu chứng)\b',
        ]
        vi_matches = sum(len(re.findall(p, text, re.I)) for p in vi_patterns)
        if vi_matches > 2 or (vi_char_count > 0 and len(text) < 100):
            return 'vi'
    
    # Default to English
    return 'en'


def _build_fallback_chat_prompt(messages, include_history: bool = True, max_history_pairs: int = 1):
    # Alpaca-style fallback prompt that works well with MedAlpaca/Gemma-derived SFTs
    # We collapse system + last user turn into an Instruction, keep brief history inline
    sys_blocks = [m.get("content", "").strip() for m in messages if m.get("role") == "system"]
    sys_text = "\n".join([b for b in sys_blocks if b])
    user_turns = [m.get("content", "").strip() for m in messages if m.get("role") == "user"]
    last_user = user_turns[-1] if user_turns else ""

    instruction = sys_text
    
    # Only include history if explicitly requested and if it's relevant
    if include_history:
        history_pairs = []
        current_q = None
        for m in messages:
            role = m.get("role")
            content = (m.get("content", "") or "").strip()
            if role == "user":
                current_q = content
            elif role == "assistant" and current_q:
                # Only include complete QA pairs with substantial content
                if len(current_q.strip()) > 10 and len(content.strip()) > 10:
                    history_pairs.append((current_q, content))
                current_q = None

        # Only include recent, relevant history (default: last 1 pair, max 2)
        if history_pairs:
            recent_pairs = history_pairs[-max_history_pairs:]
            # Only include if the last question is different from current question
            if recent_pairs and recent_pairs[-1][0] != last_user:
                history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in recent_pairs])
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

    # Prefer chat template only for models known to support it reliably.
    # For MedSwin / MedAlpaca SFTs, stick to a clean instruct fallback to avoid off-topic outputs.
    model_name_lower = (getattr(tokenizer, 'name_or_path', '') or '').lower()
    allow_template = not ("medswin" in model_name_lower or "medalpaca" in model_name_lower)
    if allow_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            chat_msgs = [{"role": "system", "content": sys}]
            for m in messages:
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
    
    # For gated models like MedGemma, always use explicit token to ensure authentication
    # This is more reliable than relying on environment variables
    max_retries = 3
    retry_delay = 2  # seconds
    
    # Try loading tokenizer with explicit token (more reliable for gated models)
    tok = None
    for attempt in range(max_retries):
        try:
            logger.info(f"Loading tokenizer for {model_name} (attempt {attempt + 1}/{max_retries})")
            tok = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
            break
        except (OSError, PermissionError, Exception) as e:
            error_str = str(e).lower()
            if "403" in str(e) or "forbidden" in error_str or "gated" in error_str:
                error_msg = (
                    f"Access denied to model '{model_name}'. This model is gated and requires special permissions.\n"
                    f"Please ensure your HF_TOKEN has access to gated repositories:\n"
                    f"1. Go to https://huggingface.co/settings/tokens\n"
                    f"2. Create or use a token with 'Read' access\n"
                    f"3. Enable 'Access to public gated repositories' in token settings\n"
                    f"4. Accept the model's terms at https://huggingface.co/{model_name}"
                )
                logger.error(error_msg)
                raise PermissionError(error_msg) from e
            elif "couldn't connect" in error_str or "connection" in error_str or "network" in error_str:
                if attempt < max_retries - 1:
                    logger.warning(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    # Try slow tokenizer as last resort before giving up
                    logger.warning(f"Fast tokenizer failed. Trying slow tokenizer as last resort...")
                    try:
                        tok = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN, use_fast=False)
                        break
                    except Exception as e2:
                        error_msg = (
                            f"Failed to connect to HuggingFace Hub to load '{model_name}'.\n"
                            f"Please check your internet connection and try again.\n"
                            f"If the model is gated, ensure your HF_TOKEN is valid and has access."
                        )
                        logger.error(error_msg)
                        raise ConnectionError(error_msg) from e2
            elif attempt < max_retries - 1:
                logger.warning(f"Error loading tokenizer (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            else:
                # Try slow tokenizer as last resort
                logger.warning(f"Fast tokenizer load failed ({e}). Retrying with slow tokenizer...")
                try:
                    tok = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN, use_fast=False)
                    break
                except Exception as e2:
                    raise
    
    if tok is None:
        raise RuntimeError(f"Failed to load tokenizer for {model_name} after {max_retries} attempts")

    if tok.eos_token_id is None and getattr(tok, "eos_token", None) is None:
        tok.eos_token = "</s>"
    if tok.pad_token_id is None and getattr(tok, "pad_token", None) is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
    if tok.bos_token_id is None and getattr(tok, "bos_token", None) is None:
        # set a reasonable BOS to stabilize some chat models
        tok.bos_token = tok.eos_token or tok.pad_token or "<s>"
    tok.padding_side = "right"

    # Load model with explicit token (more reliable for gated models)
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            logger.info(f"Loading model for {model_name} (attempt {attempt + 1}/{max_retries})")
            mdl = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True,
                token=HF_TOKEN,
                dtype=dtype,
                low_cpu_mem_usage=True,
            )
            break
        except (OSError, PermissionError, Exception) as e:
            error_str = str(e).lower()
            if "403" in str(e) or "forbidden" in error_str or "gated" in error_str:
                error_msg = (
                    f"Access denied to model '{model_name}'. This model is gated and requires special permissions.\n"
                    f"Please ensure your HF_TOKEN has access to gated repositories:\n"
                    f"1. Go to https://huggingface.co/settings/tokens\n"
                    f"2. Create or use a token with 'Read' access\n"
                    f"3. Enable 'Access to public gated repositories' in token settings\n"
                    f"4. Accept the model's terms at https://huggingface.co/{model_name}"
                )
                logger.error(error_msg)
                raise PermissionError(error_msg) from e
            elif "couldn't connect" in error_str or "connection" in error_str or "network" in error_str:
                if attempt < max_retries - 1:
                    logger.warning(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    error_msg = (
                        f"Failed to connect to HuggingFace Hub to load '{model_name}'.\n"
                        f"Please check your internet connection and try again.\n"
                        f"If the model is gated, ensure your HF_TOKEN is valid and has access."
                    )
                    logger.error(error_msg)
                    raise ConnectionError(error_msg) from e
            elif attempt < max_retries - 1:
                logger.warning(f"Error loading model (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            else:
                raise

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
    logger.info(
        "tokenizer(name=%s, has_template=%s, eos=%s, pad=%s, bos=%s)",
        getattr(tok, 'name_or_path', ''),
        hasattr(tok, 'chat_template') and bool(getattr(tok, 'chat_template', None)),
        str(tok.eos_token), str(tok.pad_token), str(tok.bos_token),
    )
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

    # --- language detection ---
    detected_lang = _detect_language(message)
    logger.info(f"[language] detected={detected_lang} for query: {message[:50]}...")
    
    # --- retrieval (optional) ---
    context = ""
    source_info = ""
    try:
        if not disable_retrieval:
            if not os.path.exists(index_dir):
                yield history + [{"role": "assistant", "content":
                    "Please upload documents first or enable 'Disable document retrieval' to chat without documents."}]
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

            # For Vietnamese queries, be more selective with context to avoid hallucinations
            # Only use context if it's highly relevant (top 3-5 nodes) and filter out irrelevant content
            if detected_lang == 'vi':
                # Use fewer, more relevant nodes for Vietnamese to reduce hallucination
                merged_nodes = merged_nodes[:min(5, len(merged_nodes))]
                logger.info(f"[retrieval] Vietnamese query - limiting to top {len(merged_nodes)} nodes")
            
            # merge text + normalize + truncate by tokens to keep headroom for generation
            context = "\n\n".join([(n.node.text or "") for n in merged_nodes])
            context = _normalize_text(context)
            
            # For Vietnamese, use less context to avoid confusion
            max_context_tokens = 1200 if detected_lang == 'vi' else 1800
            context = _truncate_by_tokens(context, global_tokenizer, max_tokens=max_context_tokens)

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
    
    # Handle system prompt based on language and whether RAG is enabled
    if detected_lang == 'vi':
        if context:
            # For Vietnamese with context, be more explicit about using only relevant context
            sys_text = f"{sys_text}\n\n[Lưu ý: Chỉ sử dụng thông tin từ ngữ cảnh tài liệu nếu nó liên quan trực tiếp đến câu hỏi.]\n\n[Document Context]\n{context}{source_info}"
        else:
            # For Vietnamese without context (RAG disabled), emphasize using medical knowledge
            if disable_retrieval:
                sys_text = f"{sys_text}\n\n[Lưu ý: Trả lời bằng tiếng Việt dựa trên kiến thức y tế của bạn.]"
            else:
                sys_text = f"{sys_text}\n\n[Lưu ý: Trả lời bằng tiếng Việt dựa trên kiến thức y tế của bạn.]"
    else:
        # For English/other languages
        if context:
            # Add context normally when RAG is enabled
            sys_text = f"{sys_text}\n\n[Document Context]\n{context}{source_info}"
        elif disable_retrieval:
            # When RAG is disabled, instruct model to use its own medical knowledge
            sys_text = f"{sys_text}\n\n[Note: Answer based on your medical knowledge.]"

    # Reconstruct conversation for template with smart history filtering
    # Only include recent, relevant history to avoid confusion and hallucinations
    # For Vietnamese queries, be even more restrictive with history
    convo_msgs = [{"role": "system", "content": sys_text}]
    
    # Filter history: only include complete QA pairs that are recent and relevant
    filtered_history = []
    if history:
        # For Vietnamese, only include last 2 messages (1 QA pair) to reduce confusion
        max_history = 2 if detected_lang == 'vi' else 4
        recent_history = history[-max_history:] if len(history) > max_history else history
        
        # Only include if messages form complete pairs and are substantial
        # For Vietnamese, also check that history is in same language
        for i, m in enumerate(recent_history):
            if m and isinstance(m, dict) and m.get("role") in ("user", "assistant"):
                content = m.get("content", "").strip()
                # Only include if content is substantial (not empty or very short)
                if len(content) > 5:
                    # For Vietnamese, only include history if it's also Vietnamese
                    if detected_lang == 'vi':
                        hist_lang = _detect_language(content)
                        if hist_lang == 'vi':
                            filtered_history.append({"role": m["role"], "content": content})
                    else:
                        filtered_history.append({"role": m["role"], "content": content})
    
    # Add filtered history
    for m in filtered_history:
        convo_msgs.append(m)
    
    # Add current message
    convo_msgs.append({"role": "user", "content": message})

    used_chat_template = False
    if hasattr(global_tokenizer, "apply_chat_template"):
        try:
            prompt = global_tokenizer.apply_chat_template(convo_msgs, tokenize=False, add_generation_prompt=True)
            used_chat_template = True
        except Exception:
            # fallback to a clean instruct format with limited history
            prompt = _build_fallback_chat_prompt(convo_msgs, include_history=len(filtered_history) > 0, max_history_pairs=1)
    else:
        # Use fallback with limited history
        prompt = _build_fallback_chat_prompt(convo_msgs, include_history=len(filtered_history) > 0, max_history_pairs=1)

    # --- streaming infra ---
    stop_event = threading.Event()

    class StopOnEvent(StoppingCriteria):
        def __init__(self, stop_event):
            super().__init__()
            self.stop_event = stop_event
        def __call__(self, input_ids, scores, **kwargs):
            return self.stop_event.is_set()
    
    class IgnoreEOSUntilMinTokens(StoppingCriteria):
        """Prevent stopping on EOS token until we reach a minimum number of tokens."""
        def __init__(self, eos_token_id, min_tokens_to_ignore_eos, prompt_length):
            super().__init__()
            self.eos_token_id = eos_token_id
            self.min_tokens_to_ignore_eos = min_tokens_to_ignore_eos
            self.prompt_length = prompt_length
        
        def __call__(self, input_ids, scores, **kwargs):
            # Calculate how many new tokens have been generated
            new_tokens_count = input_ids.shape[-1] - self.prompt_length
            
            # If we haven't reached the minimum threshold and EOS is generated, prevent stopping
            if new_tokens_count < self.min_tokens_to_ignore_eos:
                if input_ids[0, -1].item() == self.eos_token_id:
                    logger.warning(f"EOS token generated at {new_tokens_count} tokens (min: {self.min_tokens_to_ignore_eos}), but stopping criteria cannot prevent it")
            
            # Never stop based on this criteria alone - let other criteria handle stopping
            return False
    
    class PreventEOSLogitsProcessor(LogitsProcessor):
        """Prevent EOS token from being generated until we reach a minimum number of tokens.
        
        This is the actual solution - we prevent EOS tokens from being generated in the first place
        until we reach the minimum threshold. This way, the model can't stop early.
        
        Additionally, we can add logic to detect incomplete responses and prevent stopping
        even after the threshold if the response seems incomplete.
        """
        def __init__(self, eos_token_id, min_tokens_to_ignore_eos, prompt_length, max_new_tokens):
            self.eos_token_id = eos_token_id
            self.min_tokens_to_ignore_eos = min_tokens_to_ignore_eos
            self.prompt_length = prompt_length
            self.max_new_tokens = max_new_tokens
        
        def __call__(self, input_ids, scores):
            # Calculate how many new tokens have been generated
            new_tokens_count = input_ids.shape[-1] - self.prompt_length
            
            # Always prevent EOS if we haven't reached the minimum threshold
            if new_tokens_count < self.min_tokens_to_ignore_eos:
                scores[:, self.eos_token_id] = float('-inf')
                return scores
            
            # Even after reaching the threshold, be more aggressive:
            # Only allow EOS if we're very close to max_new_tokens (within 5%)
            # This ensures we use almost all available tokens
            tokens_remaining = self.max_new_tokens - new_tokens_count
            threshold_remaining = int(self.max_new_tokens * 0.05)  # 5% of max tokens
            
            if tokens_remaining > threshold_remaining:
                # Still too many tokens remaining - prevent EOS
                scores[:, self.eos_token_id] = float('-inf')
            
            return scores

    # Don't skip special tokens - they're important for proper decoding, especially for non-English languages
    # skip_special_tokens=False ensures Vietnamese and other languages decode correctly
    # Use timeout=None to ensure streamer doesn't stop prematurely
    streamer = TextIteratorStreamer(
        global_tokenizer, 
        skip_prompt=True, 
        skip_special_tokens=False,
        timeout=None  # Don't timeout - wait for all tokens
    )

    # fit prompt within context window
    ctx = int(getattr(global_model.config, "max_position_embeddings", 4096))
    max_inp = max(256, ctx - int(max_new_tokens) - 8)
    
    # Tokenize with proper handling to avoid cutting in the middle of tokens
    # This is especially important for Vietnamese and other languages with complex tokenization
    enc = global_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_inp,
        add_special_tokens=False,
        padding=False,
    )
    
    # Ensure input_ids are on the correct device
    if hasattr(enc, 'to'):
        enc = enc.to(global_model.device)
    else:
        # Handle case where enc is a dict
        enc = {k: v.to(global_model.device) if hasattr(v, 'to') else v for k, v in enc.items()}

    # avoid aggressive bad-words filtering which can distort outputs
    bad_words_ids = None

    # Honour temperature: sample iff temperature > 0
    use_sampling = float(temperature) > 0.0
    # Set minimum tokens to ensure complete responses - increased significantly to prevent premature stopping
    # Use at least 10% of max_new_tokens or 64 tokens, whichever is higher
    min_tokens = max(64, int(max_new_tokens * 0.10))

    pad_id = global_tokenizer.pad_token_id
    if pad_id is None:
        pad_id = global_tokenizer.eos_token_id
    eos_id = global_tokenizer.eos_token_id
    
    # Calculate prompt length for the stopping criteria
    prompt_length = enc['input_ids'].shape[-1] if isinstance(enc, dict) else enc.input_ids.shape[-1]
    
    # Set minimum tokens before allowing EOS to stop (use 95% of max_new_tokens or min_tokens, whichever is higher)
    # This ensures the model generates almost all tokens before EOS can stop, preventing incomplete responses
    # Using 95% instead of 100% allows for natural stopping at the very end if the model truly finishes
    min_tokens_before_eos = max(min_tokens, int(max_new_tokens * 0.95))
    
    # Create stopping criteria list with both custom criteria
    stopping_criteria = StoppingCriteriaList([
        StopOnEvent(stop_event),
        IgnoreEOSUntilMinTokens(eos_id, min_tokens_before_eos, prompt_length)
    ])
    
    # Create logits processor to prevent EOS tokens from being generated until we reach the threshold
    # This is the key fix - we prevent EOS tokens from being generated in the first place
    # Pass max_new_tokens so the processor can be more aggressive about preventing early stopping
    logits_processor = LogitsProcessorList([
        PreventEOSLogitsProcessor(eos_id, min_tokens_before_eos, prompt_length, int(max_new_tokens))
    ])
    
    # Configure stop sequences to prevent premature stopping
    # Don't stop on common mid-sentence patterns that might appear in medical text
    stop_sequences = None  # Let the model use its natural EOS token
    
    # IMPORTANT: To prevent premature EOS stopping, we use a very high min_new_tokens threshold.
    # This ensures the model generates at least 95% of max_new_tokens before EOS can stop it.
    # We keep EOS enabled because disabling it causes the model to generate indefinitely and go off-topic.
    # The min_new_tokens parameter ensures the model generates enough tokens before EOS can stop.
    # Additionally, the logits processor prevents EOS until we're very close to max_new_tokens.
    
    # Use the higher min_new_tokens threshold (95% of max_new_tokens)
    # This ensures almost all tokens are used before EOS can stop, preventing incomplete responses
    effective_min_tokens = min_tokens_before_eos
    
    # Keep EOS enabled - disabling it causes off-topic generation
    # Instead, rely on min_new_tokens to prevent early stopping
    generation_eos_token_id = eos_id  # Keep EOS enabled
    
    # Update model's generation_config to ensure our settings take effect
    if hasattr(global_model, 'generation_config'):
        global_model.generation_config.min_new_tokens = effective_min_tokens
        global_model.generation_config.max_new_tokens = int(max_new_tokens)
        # Don't override eos_token_id in generation_config - let it use the default
    
    logger.info(f"Generation config: max_new_tokens={max_new_tokens}, min_new_tokens={effective_min_tokens}, eos_token_id={generation_eos_token_id}")
    logger.info(f"EOS token enabled - model will generate at least {effective_min_tokens} tokens (95% of max) before EOS can stop")
    logger.info(f"Logits processor will prevent EOS until {min_tokens_before_eos} tokens, then allow only when <5% tokens remain")
    
    generation_kwargs = dict(
        **enc,
        streamer=streamer,
        max_new_tokens=int(max_new_tokens),
        min_new_tokens=effective_min_tokens,  # Use the higher threshold to prevent early EOS stopping
        do_sample=use_sampling,
        repetition_penalty=max(1.1, float(penalty)),
        no_repeat_ngram_size=4,
        use_cache=True,
        stopping_criteria=stopping_criteria,
        logits_processor=logits_processor,  # Add logits processor to prevent early EOS
        bad_words_ids=bad_words_ids,
        eos_token_id=generation_eos_token_id,  # Keep EOS token
        pad_token_id=pad_id,
    )
    
    # Ensure we don't stop prematurely by not setting early_stopping
    # The model should generate until max_new_tokens or natural EOS
    if use_sampling:
        generation_kwargs.update(
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
        )

    logger.info(f"chat_template={'yes' if used_chat_template else 'no'}  ctx={ctx}  max_inp={max_inp}  max_new={max_new_tokens}")
    logger.info(f"prompt_preview={(prompt[:300].replace(chr(10),' '))}")
    logger.info(f"prompt_length={len(prompt)} chars")
    # Log the ACTUAL values we're using, not the old ones
    logger.info(f"generation_config: max_new_tokens={max_new_tokens}, min_new_tokens={effective_min_tokens}, eos_token_id={generation_eos_token_id}")
    
    # Log first few tokens to verify tokenization
    try:
        test_tokens = global_tokenizer.encode(prompt[:100], add_special_tokens=False)
        logger.info(f"first_10_tokens={test_tokens[:10]}")
    except Exception as e:
        logger.warning(f"Could not preview tokens: {e}")
    
    # Start generation in a separate thread
    generation_start_time = time.time()
    thread = threading.Thread(target=global_model.generate, kwargs=generation_kwargs)
    thread.start()
    logger.info(f"Generation thread started at {generation_start_time}")

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
        # Wait for generation to complete properly
        # The streamer might stop yielding before generation completes, so we need to ensure
        # we wait for the generation thread to finish
        last_chunk_time = time.time()
        chunk_count = 0
        no_chunk_timeout = 2.0  # If no chunks for 2 seconds, check if generation is done
        
        for chunk in streamer:
            if chunk and not first_token_received.is_set():
                first_token_received.set()
            if chunk:
                partial_response += chunk
                chunk_count += 1
                last_chunk_time = time.time()
                updated_history[-1]["content"] = partial_response
                yield updated_history
        
        streamer_end_time = time.time()
        streamer_duration = streamer_end_time - generation_start_time
        logger.info(f"Streamer exhausted after {chunk_count} chunks in {streamer_duration:.2f}s. Waiting for generation thread...")
        
        # Wait for the generation thread to complete to ensure all tokens are processed
        # Use a longer timeout to ensure we capture all tokens
        # The streamer might stop yielding but generation could still be ongoing
        thread_join_timeout = 15.0  # Increased timeout for longer responses
        thread.join(timeout=thread_join_timeout)
        
        generation_end_time = time.time()
        generation_duration = generation_end_time - generation_start_time
        
        # Check if thread is still alive (didn't complete)
        if thread.is_alive():
            logger.warning(f"Generation thread still alive after {thread_join_timeout}s timeout (total: {generation_duration:.2f}s). Response may be incomplete.")
            # Force stop if it's taking too long
            stop_event.set()
            thread.join(timeout=2.0)
        else:
            logger.info(f"Generation thread completed successfully in {generation_duration:.2f}s")
        
        # Additional check: sometimes the streamer stops early but generation completes
        # Wait a bit more to ensure any final tokens are processed
        # Note: We can't iterate over the streamer again, but we can check if generation is truly done
        if not thread.is_alive():
            # Give a small delay to ensure any final buffered tokens are processed
            time.sleep(0.3)
            # The streamer should have yielded all tokens by now if generation is complete
        
        # Minimal postprocessing: trim + clean filler/disclaimers
        # Only clean if we have substantial content
        final_text = (partial_response or "").strip()
        
        # Log response length for debugging
        logger.info(f"Final response length: {len(final_text)} characters, {chunk_count} chunks")
        
        # Function to check if response is complete
        def is_response_complete(text):
            """Check if the response seems complete or incomplete."""
            if not text or len(text.strip()) < 10:
                return False
            
            text = text.strip()
            
            # Check for incomplete indicators
            incomplete_indicators = [
                text.endswith(','),
                text.endswith(';'),
                text.endswith(' and'),
                text.endswith(' or'),
                text.endswith(' but'),
                text.endswith(' with'),
                text.endswith(' for'),
                text.endswith(' to'),
                text.endswith(' the'),
                # Check if last sentence is incomplete (no period, exclamation, or question mark)
                len(text) > 50 and not any(text.rstrip().endswith(p) for p in ['.', '!', '?', ':', '\n'])
            ]
            
            # Also check if response seems to be cut off mid-word or mid-sentence
            # Look for patterns like ending with lowercase letters followed by nothing
            if len(text) > 100:
                text_rstrip = text.rstrip()
                if len(text_rstrip) > 0:
                    last_char = text_rstrip[-1]
                    # If ends with lowercase and no punctuation, might be incomplete
                    if last_char not in ['.', '!', '?', ':', '\n', ')', ']', '}']:
                        # Check if it looks like mid-sentence
                        if not last_char.isupper() and not any(incomplete_indicators):
                            # Might be incomplete if it doesn't end with proper punctuation
                            return False
            
            # Check if answer structure/plan seems complete
            # Look for patterns that suggest the answer is still in progress
            structure_indicators = [
                '1.', '2.', '3.', '4.', '5.',  # Numbered lists
                'first', 'second', 'third', 'fourth', 'fifth',  # Ordinal numbers
                'next', 'then', 'finally', 'lastly',  # Sequence indicators
                'also', 'additionally', 'furthermore', 'moreover'  # Continuation words
            ]
            
            # If text ends with a structure indicator or continuation word, might be incomplete
            last_50_chars = text[-50:].lower()
            if any(indicator in last_50_chars for indicator in structure_indicators):
                # Check if it's actually ending with these words (incomplete)
                last_words = text.split()[-3:]
                if any(word.lower() in structure_indicators for word in last_words):
                    # Ending with structure indicator - likely incomplete
                    return False
            
            # Check if response mentions incomplete sections
            incomplete_section_indicators = [
                'will be discussed', 'will be covered', 'will be explained',
                'to be continued', 'more on this', 'further details'
            ]
            
            if any(indicator in text.lower()[-100:] for indicator in incomplete_section_indicators):
                return False
            
            return not any(incomplete_indicators)
        
        # Function to extract answer structure/plan from original response
        def extract_answer_structure(text):
            """Extract the structure/plan of the answer (headings, numbered lists, etc.)."""
            if not text:
                return []
            
            structure = []
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                # Detect headings (lines starting with numbers, letters, or common heading patterns)
                if line and (line[0].isdigit() or line[0].isupper() or 
                            line.startswith('#') or line.startswith('**') or
                            any(line.startswith(prefix) for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '*'])):
                    # Extract first few words as structure element
                    words = line.split()[:5]
                    if words:
                        structure.append(' '.join(words).lower())
            
            return structure
        
        # Function to detect if continuation is off-topic or hallucinating
        def is_continuation_off_topic(original_text, continuation_text, original_message):
            """Detect if continuation text is off-topic or hallucinating."""
            if not continuation_text or len(continuation_text.strip()) < 20:
                return False
            
            continuation_lower = continuation_text.lower()
            original_lower = original_text.lower()
            message_lower = original_message.lower()
            
            # Extract key medical/context words from original message and response
            medical_keywords = [
                'medical', 'health', 'treatment', 'patient', 'symptom', 'diagnosis',
                'medication', 'therapy', 'clinical', 'disease', 'condition', 'disorder',
                'migraine', 'headache', 'pain', 'chronic', 'acute', 'preventive',
                'pharmacological', 'dosage', 'side effect', 'contraindication', 'drug',
                'prescription', 'dosage', 'mg', 'tablet', 'capsule', 'injection',
                'preventive', 'prophylactic', 'abortive', 'rescue', 'maintenance'
            ]
            
            # Extract topic keywords from original message
            message_words = set(message_lower.split())
            original_words = set(original_lower.split()[:100])  # First 100 words
            
            # Check if continuation contains medical keywords from original context
            continuation_has_medical = any(keyword in continuation_lower for keyword in medical_keywords)
            original_has_medical = any(keyword in original_lower for keyword in medical_keywords)
            
            # Check topic coherence - continuation should share some keywords with original
            continuation_words = set(continuation_lower.split()[:50])  # First 50 words
            shared_medical_words = continuation_words.intersection(set(medical_keywords))
            shared_message_words = continuation_words.intersection(message_words)
            
            # Red flags for hallucination/off-topic:
            hallucination_indicators = [
                # Wikipedia-style content
                'wikipedia:', 'de.wikipedia.org', 'https://', 'http://', 'www.',
                # Non-medical topics that shouldn't appear in medical responses
                'kawaii', 'japanisch', 'deutsch', 'anime', 'manga', 'cosplay',
                'barbie', 'hello-kitty', 'disney', 'fashion', 'pop culture',
                # Meta content about editing articles
                'edit this page', 'create one yourself', 'click below',
                'see also:', 'references', 'external links', 'commons:',
                'wikidata', 'wikitionary', 'beautywiki',
                # URL patterns
                'time.com', 'britannica.com', 'globalvoices.org', 'spiegel.de',
                # Language switching (if original was English, continuation shouldn't be German/Japanese)
                (message_lower and not any(lang in continuation_lower[:100] for lang in ['japanisch', 'deutsch', 'kawaii']) 
                 and any(lang in continuation_lower[:100] for lang in ['japanisch', 'deutsch']) 
                 and not any(med in continuation_lower[:100] for med in medical_keywords))
            ]
            
            # If continuation has hallucination indicators and no medical context, it's off-topic
            has_hallucination_indicators = any(indicator for indicator in hallucination_indicators if isinstance(indicator, str) and indicator in continuation_lower)
            has_language_switch = any(isinstance(ind, bool) and ind for ind in hallucination_indicators)
            
            if (has_hallucination_indicators or has_language_switch) and not continuation_has_medical:
                return True
            
            # If original had medical context but continuation doesn't, might be off-topic
            if original_has_medical and not continuation_has_medical and len(continuation_text) > 100:
                # Check if continuation is about completely different topics
                non_medical_topics = ['fashion', 'anime', 'manga', 'culture', 'internet', 'viral', 'trend', 'kawaii']
                if any(topic in continuation_lower for topic in non_medical_topics):
                    return True
            
            # Check topic coherence - if continuation has no shared medical/message words, might be off-topic
            if original_has_medical and len(continuation_text) > 100:
                if len(shared_medical_words) == 0 and len(shared_message_words) == 0:
                    # No shared keywords at all - likely off-topic
                    return True
            
            # Check if continuation seems to be repeating or going in circles
            if len(continuation_text) > 200:
                # Check for excessive repetition of same phrases
                words = continuation_lower.split()
                if len(words) > 50:
                    word_counts = {}
                    for word in words:
                        if len(word) > 4:  # Only check longer words
                            word_counts[word] = word_counts.get(word, 0) + 1
                    # If any word appears more than 10 times in 200 chars, might be stuck
                    if any(count > 10 for count in word_counts.values()):
                        return True
            
            return False
        
        # Function to check if continuation is completing the answer plan
        def is_continuation_on_plan(original_text, continuation_text, original_message):
            """Check if continuation is actually continuing the original answer plan."""
            if not continuation_text or len(continuation_text.strip()) < 20:
                return False
            
            # Extract structure from original response
            original_structure = extract_answer_structure(original_text)
            
            # Check if continuation mentions topics from original structure
            continuation_lower = continuation_text.lower()
            
            # If original had structure, continuation should relate to it
            if original_structure:
                structure_mentions = sum(1 for struct_elem in original_structure if struct_elem in continuation_lower)
                if structure_mentions == 0 and len(continuation_text) > 100:
                    # Continuation doesn't mention any original structure elements
                    return False
            
            # Check if continuation is actually continuing (not starting a new topic)
            # Look for continuation indicators
            continuation_indicators = [
                'continuing', 'further', 'additionally', 'moreover', 'also',
                'next', 'another', 'in addition', 'furthermore', 'besides'
            ]
            
            # Check if continuation starts with a new topic (bad) vs continuing (good)
            first_50_chars = continuation_lower[:50]
            starts_new_topic = any(first_50_chars.startswith(new_topic) for new_topic in 
                                  ['kawaii', 'wikipedia', 'japanisch', 'deutsch', 'anime'])
            
            if starts_new_topic:
                return False
            
            return True
        
        # Check if response is complete
        is_complete = is_response_complete(final_text)
        
        # Sequential generation: continue generating if response is incomplete
        # This allows the model to go beyond max_new_tokens to complete its answer
        max_total_tokens = int(max_new_tokens * 3)  # Allow up to 3x max_new_tokens total
        continuation_chunk_size = int(max_new_tokens * 0.5)  # Generate 50% more tokens each continuation
        total_tokens_generated = chunk_count  # Approximate from chunks
        continuation_count = 0
        max_continuations = 3  # Reduced from 5 to prevent excessive hallucination
        original_response_before_continuation = final_text  # Store original response
        
        while not is_complete and continuation_count < max_continuations and total_tokens_generated < max_total_tokens:
            continuation_count += 1
            logger.info(f"Response incomplete - continuing generation (continuation {continuation_count}/{max_continuations})")
            
            # Prepare continuation prompt more carefully to maintain context and answer plan
            # Extract the answer structure/plan from original response
            answer_structure = extract_answer_structure(original_response_before_continuation)
            
            # Use the last portion of the response to maintain context
            # But also include key structure elements to maintain the plan
            response_tail = final_text[-400:] if len(final_text) > 400 else final_text  # Last 400 chars
            
            # Build continuation prompt with explicit instructions to maintain plan
            structure_hint = ""
            if answer_structure:
                structure_hint = f"\n[Answer structure/plan to continue: {', '.join(answer_structure[:3])}...]"
            
            continuation_prompt = (
                f"{prompt}\n\n"
                f"[Previous response (incomplete):]\n{response_tail}\n\n"
                f"{structure_hint}\n\n"
                f"[Instructions: Continue the medical response from where it left off. "
                f"Stay on the same topic and complete the answer plan. "
                f"Do NOT start new topics or include Wikipedia-style content, URLs, or non-medical topics. "
                f"Continue naturally from the last sentence.]"
            )
            
            # Tokenize continuation prompt
            continuation_enc = global_tokenizer(
                continuation_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_inp,  # Use same max input length
                add_special_tokens=False,
                padding=False,
            )
            
            # Ensure input_ids are on the correct device
            if hasattr(continuation_enc, 'to'):
                continuation_enc = continuation_enc.to(global_model.device)
            else:
                continuation_enc = {k: v.to(global_model.device) if hasattr(v, 'to') else v for k, v in continuation_enc.items()}
            
            # Calculate continuation prompt length for logits processor
            continuation_prompt_length = continuation_enc['input_ids'].shape[-1] if isinstance(continuation_enc, dict) else continuation_enc.input_ids.shape[-1]
            
            # Create new logits processor for continuation with correct prompt length
            continuation_logits_processor = LogitsProcessorList([
                PreventEOSLogitsProcessor(eos_id, int(continuation_chunk_size * 0.8), continuation_prompt_length, continuation_chunk_size)
            ])
            
            # Create new streamer for continuation
            continuation_streamer = TextIteratorStreamer(
                global_tokenizer, 
                skip_prompt=True, 
                skip_special_tokens=False,
                timeout=None
            )
            
            # Create continuation generation kwargs
            continuation_generation_kwargs = dict(
                **continuation_enc,
                streamer=continuation_streamer,
                max_new_tokens=continuation_chunk_size,
                min_new_tokens=int(continuation_chunk_size * 0.8),  # 80% of continuation chunk
                do_sample=use_sampling,
                repetition_penalty=max(1.1, float(penalty)),
                no_repeat_ngram_size=4,
                use_cache=True,
                stopping_criteria=StoppingCriteriaList([StopOnEvent(stop_event)]),
                logits_processor=continuation_logits_processor,  # Use continuation-specific logits processor
                bad_words_ids=bad_words_ids,
                eos_token_id=generation_eos_token_id,
                pad_token_id=pad_id,
            )
            
            if use_sampling:
                continuation_generation_kwargs.update(
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                )
            
            # Start continuation generation
            continuation_thread = threading.Thread(
                target=global_model.generate, 
                kwargs=continuation_generation_kwargs
            )
            continuation_thread.start()
            
            # Collect continuation tokens with real-time hallucination detection
            continuation_text = ""
            continuation_chunk_count = 0
            off_topic_detected = False
            
            for continuation_chunk in continuation_streamer:
                if continuation_chunk:
                    continuation_text += continuation_chunk
                    continuation_chunk_count += 1
                    
                    # Check for hallucination and plan coherence after collecting some text
                    if len(continuation_text) > 50 and continuation_chunk_count % 10 == 0:
                        # Check for off-topic/hallucination
                        if is_continuation_off_topic(original_response_before_continuation, continuation_text, message):
                            logger.warning(f"Hallucination/off-topic detected in continuation {continuation_count} after {len(continuation_text)} chars")
                            logger.warning(f"Off-topic text sample: '{continuation_text[:200]}...'")
                            off_topic_detected = True
                            # Stop the continuation thread
                            stop_event.set()
                            break
                        
                        # Check if continuation is following the plan
                        if not is_continuation_on_plan(original_response_before_continuation, continuation_text, message):
                            logger.warning(f"Continuation {continuation_count} may not be following the answer plan after {len(continuation_text)} chars")
                            # Don't stop immediately, but log warning
                    
                    # Update UI with continuation
                    final_text = (partial_response + continuation_text).strip()
                    updated_history[-1]["content"] = final_text
                    yield updated_history
            
            # Wait for continuation thread to complete
            continuation_thread.join(timeout=10.0)
            
            # If off-topic was detected, discard the continuation and stop
            if off_topic_detected:
                logger.warning(f"Stopping continuation {continuation_count} due to hallucination/off-topic content")
                # Revert to original response before continuation
                final_text = original_response_before_continuation
                partial_response = original_response_before_continuation
                is_complete = True  # Mark as complete to stop further continuations
                break
            
            # Append continuation to response only if it's on-topic
            partial_response += continuation_text
            final_text = partial_response.strip()
            chunk_count += continuation_chunk_count
            total_tokens_generated += continuation_chunk_count
            
            # Final checks for hallucination and plan coherence on the full continuation
            if is_continuation_off_topic(original_response_before_continuation, continuation_text, message):
                logger.warning(f"Hallucination detected in full continuation {continuation_count} - discarding")
                # Revert to original response
                final_text = original_response_before_continuation
                partial_response = original_response_before_continuation
                is_complete = True
                break
            
            # Check if continuation is following the plan
            if not is_continuation_on_plan(original_response_before_continuation, continuation_text, message):
                logger.warning(f"Continuation {continuation_count} may not be following the answer plan - checking if acceptable")
                # If it's not following the plan but also not off-topic, we might still accept it
                # but log a warning
                if len(continuation_text) > 200 and not continuation_text.lower().startswith(('kawaii', 'wikipedia', 'japanisch')):
                    # If it's substantial and not obviously wrong, accept it but warn
                    logger.warning(f"Accepting continuation {continuation_count} despite plan deviation")
                else:
                    # If it's clearly wrong, discard it
                    logger.warning(f"Discarding continuation {continuation_count} - not following plan")
                    final_text = original_response_before_continuation
                    partial_response = original_response_before_continuation
                    is_complete = True
                    break
            
            # Check if continuation is complete
            is_complete = is_response_complete(final_text)
            
            logger.info(f"Continuation {continuation_count} complete: {len(continuation_text)} chars, {continuation_chunk_count} chunks. Total: {len(final_text)} chars. Complete: {is_complete}")
            
            # If we got very few tokens in continuation, might be done
            if continuation_chunk_count < 10:
                logger.info("Continuation produced very few tokens, assuming complete")
                is_complete = True
        
        if continuation_count > 0:
            logger.info(f"Sequential generation complete after {continuation_count} continuation(s). Total response: {len(final_text)} chars")
        
        if not is_complete:
            logger.warning(f"Response may still be incomplete after {continuation_count} continuation(s) - ends with: '{final_text[-50:]}'")
        
        if len(final_text) > 10:  # Only clean if we have meaningful content
            # Remove any special token strings that might have appeared (e.g., <|endoftext|>, </s>, etc.)
            # This is safe because we preserved special tokens during decoding for proper language handling
            special_token_strings = [
                "<|endoftext|>", "</s>", "<s>", "<|pad|>", 
                "<|im_start|>", "<|im_end|>", "<|user|>", "<|assistant|>"
            ]
            for token_str in special_token_strings:
                final_text = final_text.replace(token_str, "")
            final_text = _clean_leading_filler(_strip_disclaimers(final_text))
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
                    choices=["MedSwin-7B KD", "MedSwin-7B SFT", "MedAlpaca-7B", "MedGemma-27B"],
                    value="MedSwin-7B KD",
                    label="Model"
                )
                model_status = gr.Textbox(
                    label="Model Status",
                    placeholder="Select a model...",
                    interactive=False,
                    visible=True
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
                            "Answer clinically and concisely using the provided context. "
                            "If context is insufficient, state what is missing. Cite filenames in brackets when used."
                        ),
                        label="System Prompt",
                        lines=3
                    )
                    gr.Markdown(
                        "**Clinical Use Disclaimer:** This application is intended for informational purposes only and does not constitute medical advice. "
                        "Always consult qualified healthcare professionals for diagnosis and treatment decisions.")
                    
                    with gr.Tab("Generation Parameters"):
                        temperature = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=0.1,
                            label="Temperature"
                        )
                        max_new_tokens = gr.Slider(
                            minimum=64,
                            maximum=1024,
                            step=32,
                            value=768,
                            label="Max New Tokens",
                        )
                        top_p = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=0.5,
                            label="Top P"
                        )
                        top_k = gr.Slider(
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=20,
                            label="Top K"
                        )
                        penalty = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            step=0.1,
                            value=1.3,
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
                    try:
                        if choice == "MedSwin-7B KD":
                            name = MEDSWIN_KD_MODEL
                        elif choice == "MedSwin-7B SFT":
                            name = MEDSWIN_SFT_MODEL
                        elif choice == "MedAlpaca-7B":
                            name = MEDALPACA_MODEL
                        else:  # MedGemma-27B
                            name = MEDGEMMA_MODEL
                        initialize_model_and_tokenizer(name)
                        return f"Loaded: {choice}"
                    except PermissionError as e:
                        error_msg = str(e)
                        logger.error(f"Model loading failed: {error_msg}")
                        return f"Error: {error_msg}"
                    except ConnectionError as e:
                        error_msg = str(e)
                        logger.error(f"Model loading failed: {error_msg}")
                        return f"Connection Error: {error_msg}"
                    except Exception as e:
                        error_msg = f"Failed to load model: {str(e)}"
                        logger.error(f"Model loading failed: {error_msg}")
                        return f"Error: {error_msg}"

                submit_button.click(
                    fn=stream_chat,
                    inputs=[message_input, chatbot, system_prompt, disable_retrieval,
                            temperature, max_new_tokens, top_p, top_k, penalty,
                            retriever_k, merge_threshold],
                    outputs=chatbot
                )

                message_input.submit(
                    fn=stream_chat,
                    inputs=[message_input, chatbot, system_prompt, disable_retrieval,
                            temperature, max_new_tokens, top_p, top_k, penalty,
                            retriever_k, merge_threshold],
                    outputs=chatbot
                )

                model_selector.change(
                    fn=_on_model_change,
                    inputs=[model_selector],
                    outputs=[model_status]
                )
    return demo


if __name__ == "__main__":
    initialize_model_and_tokenizer()
    demo = create_demo()
    demo.launch()