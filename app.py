import gradio as gr
import os
import PyPDF2
import logging
import torch
import threading
import time
import re, unicodedata
import numpy as np
import json
import hashlib
from pathlib import Path

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

# Retry configuration for ZeroGPU timeout handling
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds between retries

# Cache configuration for GPU abort recovery
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_EXPIRY = 3600  # Cache expires after 1 hour

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
global_embedding_model = None
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

def initialize_embedding_model():
    """Initialize the embedding model at startup."""
    global global_embedding_model
    if global_embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        try:
            global_embedding_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL, token=HF_TOKEN)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise


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


def _get_cache_key(message: str, history: list, params: dict) -> str:
    """Generate a unique cache key for a chat request."""
    # Create a hash from message, history, and parameters
    cache_data = {
        'message': message,
        'history': history,
        'params': params
    }
    cache_str = json.dumps(cache_data, sort_keys=True)
    cache_hash = hashlib.sha256(cache_str.encode()).hexdigest()
    return cache_hash


def _save_cache(cache_key: str, partial_response: str, continuation_count: int = 0, metadata: dict = None):
    """Save partial response to cache for recovery."""
    try:
        cache_file = CACHE_DIR / f"{cache_key}.json"
        cache_data = {
            'partial_response': partial_response,
            'continuation_count': continuation_count,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Cache saved: {cache_key[:16]}... ({len(partial_response)} chars)")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")


def _load_cache(cache_key: str) -> dict:
    """Load cached partial response if available and not expired."""
    try:
        cache_file = CACHE_DIR / f"{cache_key}.json"
        if not cache_file.exists():
            return None
        
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # Check if cache is expired
        age = time.time() - cache_data.get('timestamp', 0)
        if age > CACHE_EXPIRY:
            logger.info(f"Cache expired: {cache_key[:16]}... (age: {age:.0f}s)")
            cache_file.unlink()  # Delete expired cache
            return None
        
        logger.info(f"Cache loaded: {cache_key[:16]}... ({len(cache_data.get('partial_response', ''))} chars, age: {age:.0f}s)")
        return cache_data
    except Exception as e:
        logger.warning(f"Failed to load cache: {e}")
        return None


def _delete_cache(cache_key: str):
    """Delete cache file after successful completion."""
    try:
        cache_file = CACHE_DIR / f"{cache_key}.json"
        if cache_file.exists():
            cache_file.unlink()
            logger.info(f"Cache deleted: {cache_key[:16]}...")
    except Exception as e:
        logger.warning(f"Failed to delete cache: {e}")


def _cleanup_old_cache():
    """Clean up expired cache files."""
    try:
        current_time = time.time()
        cleaned = 0
        for cache_file in CACHE_DIR.glob("*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                age = current_time - cache_data.get('timestamp', 0)
                if age > CACHE_EXPIRY:
                    cache_file.unlink()
                    cleaned += 1
            except Exception:
                # If file is corrupted or can't be read, delete it
                cache_file.unlink()
                cleaned += 1
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} expired cache files")
    except Exception as e:
        logger.warning(f"Failed to cleanup cache: {e}")


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


@spaces.GPU(max_duration=120)
def create_or_update_index(files, request: gr.Request):
    """Create or update index with retry logic for ZeroGPU timeouts."""
    global global_file_info
    
    if not files:
        return "Please provide files.", ""
    
    # Retry logic for ZeroGPU timeout handling
    last_exception = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _create_or_update_index_impl(files, request)
        except KeyboardInterrupt:
            # User interrupted, don't retry
            raise
        except Exception as e:
            last_exception = e
            error_msg = str(e).lower()
            # Check if it's a timeout/abort related error
            is_timeout = any(keyword in error_msg for keyword in [
                'timeout', 'abort', 'exceeded', 'duration', 'max_duration',
                'killed', 'terminated', 'interrupted', 'connection reset'
            ])
            
            if is_timeout and attempt < MAX_RETRIES:
                logger.warning(f"Index creation attempt {attempt}/{MAX_RETRIES} timed out/aborted. Retrying in {RETRY_DELAY}s...")
                logger.warning(f"Error: {e}")
                time.sleep(RETRY_DELAY)
                continue
            else:
                # Not a timeout or max retries reached
                logger.error(f"Index creation failed after {attempt} attempt(s): {e}")
                if attempt >= MAX_RETRIES:
                    return f"Error: Failed after {MAX_RETRIES} attempts. Last error: {str(e)[:200]}", ""
                raise
    
    # Should not reach here, but handle it
    if last_exception:
        return f"Error: {str(last_exception)[:200]}", ""
    return "Unknown error occurred.", ""


def _create_or_update_index_impl(files, request: gr.Request):
    """Internal implementation of create_or_update_index without retry logic."""
    global global_file_info
    
    start_time = time.time()
    user_id = request.session_hash
    save_dir = f"./{user_id}_index"
    # Initialize LlamaIndex modules
    llm = get_llm(model_name=MODEL)
    # Use global embedding model (loaded at startup)
    global global_embedding_model
    if global_embedding_model is None:
        initialize_embedding_model()
    embed_model = global_embedding_model
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


@spaces.GPU(max_duration=120)
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
    """Stream chat with retry logic and caching for GPU abort recovery."""
    # --- guards & basics ---
    if not request:
        yield history + [{"role": "assistant", "content": "Session initialization failed. Please refresh the page."}]
        return
    
    # Clean up old cache files periodically
    if hasattr(stream_chat, '_last_cleanup'):
        if time.time() - stream_chat._last_cleanup > 300:  # Every 5 minutes
            _cleanup_old_cache()
            stream_chat._last_cleanup = time.time()
    else:
        stream_chat._last_cleanup = time.time()
        _cleanup_old_cache()
    
    # Generate cache key for this request
    params = {
        'system_prompt': system_prompt,
        'disable_retrieval': disable_retrieval,
        'temperature': temperature,
        'max_new_tokens': max_new_tokens,
        'top_p': top_p,
        'top_k': top_k,
        'penalty': penalty,
        'retriever_k': retriever_k,
        'merge_threshold': merge_threshold
    }
    cache_key = _get_cache_key(message, history, params)
    
    # Try to load cached partial response
    cached_data = _load_cache(cache_key)
    resume_from_cache = cached_data is not None
    
    # Retry logic for ZeroGPU timeout handling
    last_exception = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # If we have cached data and this is a retry, resume from cache
            if resume_from_cache and attempt > 1:
                logger.info(f"Resuming from cache (attempt {attempt}/{MAX_RETRIES})")
                cached_partial = cached_data.get('partial_response', '')
                cached_continuation = cached_data.get('continuation_count', 0)
                cached_metadata = cached_data.get('metadata', {})
                
                # Yield cached partial response first
                updated_history = history + [{"role": "user", "content": message}]
                if cached_partial:
                    updated_history.append({"role": "assistant", "content": cached_partial})
                    yield updated_history
                
                # Continue generation from cache
                for result in _stream_chat_impl(
                    message, history, system_prompt, disable_retrieval,
                    temperature, max_new_tokens, top_p, top_k, penalty,
                    retriever_k, merge_threshold, request,
                    resume_from_cache=cached_partial,
                    resume_continuation_count=cached_continuation,
                    cache_key=cache_key
                ):
                    yield result
            else:
                # Normal generation or first attempt
                for result in _stream_chat_impl(
                    message, history, system_prompt, disable_retrieval,
                    temperature, max_new_tokens, top_p, top_k, penalty,
                    retriever_k, merge_threshold, request,
                    cache_key=cache_key
                ):
                    yield result
            
            # If we successfully completed, delete cache and return
            _delete_cache(cache_key)
            return
        except (GeneratorExit, KeyboardInterrupt):
            # Generator was closed by client or user interrupted, don't retry
            # But keep cache for potential recovery
            raise
        except Exception as e:
            last_exception = e
            error_msg = str(e).lower()
            # Check if it's a timeout/abort/GPU related error
            is_timeout = any(keyword in error_msg for keyword in [
                'timeout', 'abort', 'exceeded', 'duration', 'max_duration',
                'killed', 'terminated', 'interrupted', 'connection reset',
                'generator', 'generator exit', 'gpu', 'cuda', 'device',
                'out of memory', 'oom', 'cuda error', 'gpu error',
                'aborted', 'aborting', 'failed', 'error'
            ])
            
            if is_timeout and attempt < MAX_RETRIES:
                logger.warning(f"Chat generation attempt {attempt}/{MAX_RETRIES} timed out/aborted. Retrying in {RETRY_DELAY}s...")
                logger.warning(f"Error: {e}")
                # Try to load cache for next retry
                cached_data = _load_cache(cache_key)
                resume_from_cache = cached_data is not None
                if resume_from_cache:
                    logger.info(f"Cache available for retry: {len(cached_data.get('partial_response', ''))} chars")
                
                # Yield error message to user
                retry_msg = f"*[Generation interrupted. {'Resuming from cache...' if resume_from_cache else 'Retrying'} ({attempt}/{MAX_RETRIES})...]*"
                yield history + [{"role": "user", "content": message}, {"role": "assistant", "content": retry_msg}]
                time.sleep(RETRY_DELAY)
                continue
            else:
                # Not a timeout or max retries reached
                logger.error(f"Chat generation failed after {attempt} attempt(s): {e}")
                if attempt >= MAX_RETRIES:
                    # Try to return cached partial response if available
                    cached_data = _load_cache(cache_key)
                    if cached_data and cached_data.get('partial_response'):
                        logger.info(f"Returning cached partial response after {MAX_RETRIES} failed attempts")
                        cached_partial = cached_data.get('partial_response', '')
                        error_content = f"{cached_partial}\n\n*[Generation failed after {MAX_RETRIES} attempts. Partial response above.]*"
                    else:
                        error_content = f"Error: Generation failed after {MAX_RETRIES} attempts. Last error: {str(e)[:200]}"
                    yield history + [{"role": "user", "content": message}, {"role": "assistant", "content": error_content}]
                    return
                raise
    
    # Should not reach here, but handle it
    if last_exception:
        error_content = f"Error: {str(last_exception)[:200]}"
        yield history + [{"role": "user", "content": message}, {"role": "assistant", "content": error_content}]
    else:
        yield history + [{"role": "user", "content": message}, {"role": "assistant", "content": "Unknown error occurred."}]


def _stream_chat_impl(
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
    request: gr.Request,
    cache_key: str = None,
    resume_from_cache: str = None,
    resume_continuation_count: int = 0
):
    """Internal implementation of stream_chat with caching support."""
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

            # Use global embedding model (loaded at startup)
            global global_embedding_model
            if global_embedding_model is None:
                initialize_embedding_model()
            embed_model = global_embedding_model
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
    
    # Handle resume from cache if available
    if resume_from_cache:
        logger.info(f"Resuming from cache: {len(resume_from_cache)} chars, continuation: {resume_continuation_count}")
        partial_response = resume_from_cache
        final_text = partial_response
        chunk_count = 0  # No chunks generated yet
        # Update history with cached response
        updated_history = history + [{"role": "user", "content": message}]
        if partial_response:
            updated_history.append({"role": "assistant", "content": partial_response})
            yield updated_history
        # Skip initial generation and go straight to continuation logic
        skip_initial_generation = True
    else:
        # Start generation in a separate thread
        generation_start_time = time.time()
        thread = threading.Thread(target=global_model.generate, kwargs=generation_kwargs)
        thread.start()
        logger.info(f"Generation thread started at {generation_start_time}")

        # prime UI
        updated_history = (history or []) + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
        yield updated_history

        partial_response = ""
        skip_initial_generation = False
        
        first_token_received = threading.Event()

        def _watch_first_token():
            if not first_token_received.wait(timeout=45):
                logger.warning("Generation timeout: no tokens in 45s; stopping stream.")
                stop_event.set()

        watchdog = threading.Thread(target=_watch_first_token, daemon=True)
        watchdog.start()

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
                
                # Save cache periodically (every 50 chunks to avoid overhead)
                if cache_key and chunk_count % 50 == 0:
                    _save_cache(cache_key, partial_response, 0, {'chunk_count': chunk_count})
                
                yield updated_history
        
        if not skip_initial_generation:
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
            
            # Save cache after initial generation
            if cache_key and final_text:
                _save_cache(cache_key, final_text, 0, {'chunk_count': chunk_count, 'stage': 'initial'})
            
            # Log response length for debugging
            logger.info(f"Final response length: {len(final_text)} characters, {chunk_count} chunks")
        else:
            # When resuming from cache, initialize continuation_count from cache
            continuation_count = resume_continuation_count
            # Mark response as incomplete to continue from where we left off
            is_complete = False
        
        # Function to check if response is complete
        def is_response_complete(text):
            """Check if the response seems complete or incomplete.
            
            This should be accurate - check for real incomplete patterns,
            not just punctuation. A response can end with punctuation but still be incomplete.
            """
            if not text or len(text.strip()) < 10:
                return False
            
            text = text.strip()
            
            # Check for incomplete indicators (these always indicate incompleteness)
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
                len(text) > 50 and not any(text.rstrip().endswith(p) for p in ['.', '!', '?', ':', '\n', ')', ']', '}'])
            ]
            
            # If any incomplete indicator is present, definitely incomplete
            if any(incomplete_indicators):
                return False
            
            # Check if response seems to be cut off mid-word or mid-sentence
            if len(text) > 100:
                text_rstrip = text.rstrip()
                if len(text_rstrip) > 0:
                    last_char = text_rstrip[-1]
                    # If ends with lowercase and no punctuation, might be incomplete
                    if last_char not in ['.', '!', '?', ':', '\n', ')', ']', '}']:
                        # Check if it looks like mid-sentence
                        if not last_char.isupper():
                            # Might be incomplete if it doesn't end with proper punctuation
                            return False
            
            # Check if answer structure/plan seems complete
            # Look for patterns that suggest the answer is still in progress
            structure_indicators = [
                '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.',  # Numbered lists
                'first', 'second', 'third', 'fourth', 'fifth',  # Ordinal numbers
                'next', 'then', 'finally', 'lastly',  # Sequence indicators
                'also', 'additionally', 'furthermore', 'moreover'  # Continuation words
            ]
            
            # Check if text ends with a structure indicator or continuation word
            last_100_chars = text[-100:].lower()
            last_words = text.split()[-5:]  # Check last 5 words
            
            # If ends with structure indicator, likely incomplete
            for word in last_words:
                if word.lower().rstrip('.,;:!?') in structure_indicators:
                    # Ending with structure indicator - likely incomplete
                    return False
            
            # Check for incomplete list items (ending with bullet points or numbers)
            if text.endswith(('*', '-', '•', '1.', '2.', '3.', '4.', '5.')):
                return False
            
            # Check if response mentions incomplete sections
            incomplete_section_indicators = [
                'will be discussed', 'will be covered', 'will be explained',
                'to be continued', 'more on this', 'further details',
                'more information', 'additional details'
            ]
            
            if any(indicator in text.lower()[-150:] for indicator in incomplete_section_indicators):
                return False
            
            # Check for incomplete HTML/list structures (like ending with </li> or <ul>)
            if text.rstrip().endswith(('</li>', '<ul>', '<ol>', '<li>', '</ul>', '</ol>')):
                return False
            
            # Check if response ends mid-word (very short last word suggests cutoff)
            last_word = text.split()[-1] if text.split() else ''
            if len(last_word) < 3 and len(text) > 100:
                # Very short last word might indicate cutoff
                return False
            
            # If we get here, response seems complete
            # But only if it ends with proper punctuation AND seems finished
            if any(text.rstrip().endswith(p) for p in ['.', '!', '?', ':', '\n', ')', ']', '}']):
                # Additional check: make sure it's not ending mid-sentence
                # Check if last sentence is complete (has subject and verb)
                last_sentence = text.split('.')[-1].strip() if '.' in text else text.split()[-10:]
                if isinstance(last_sentence, list):
                    last_sentence = ' '.join(last_sentence)
                
                # If last sentence is very short (<10 chars), might be incomplete
                if len(last_sentence) < 10:
                    return False
                
                # If last sentence doesn't have proper structure, might be incomplete
                # Check if it has at least 3 words (subject, verb, object)
                if len(last_sentence.split()) < 3:
                    return False
                
                return True
            
            # If no proper punctuation, definitely incomplete
            return False
        
        # Function to extract answer structure/plan from original response
        def extract_answer_structure(text):
            """Extract the structure/plan of the answer (headings, numbered lists, etc.)."""
            if not text:
                return []
            
            structure = []
            lines = text.split('\n')
            
            # Track numbering sequence to detect incomplete lists
            expected_number = 1
            last_number = 0
            numbered_items = []  # Track numbered items separately
            
            for line in lines:
                line = line.strip()
                # Detect headings (lines starting with numbers, letters, or common heading patterns)
                if line and (line[0].isdigit() or line[0].isupper() or 
                            line.startswith('#') or line.startswith('**') or
                            any(line.startswith(prefix) for prefix in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '-', '*'])):
                    # Extract first few words as structure element
                    words = line.split()[:5]
                    if words:
                        structure_item = ' '.join(words).lower()
                        structure.append(structure_item)
                        
                        # Track numbering for sequence detection
                        if line[0].isdigit():
                            try:
                                num = int(line.split('.')[0])
                                numbered_items.append((num, structure_item))
                                if num > last_number:
                                    last_number = num
                                    expected_number = num + 1
                            except:
                                pass
            
            # If we detected a numbered list, check if it's complete
            if last_number > 0 and last_number < expected_number:
                # List might be incomplete - add expected next item to structure
                structure.append(f"item {expected_number}")
            
            # Return structure with numbering info
            return {
                'items': structure,
                'numbered_items': numbered_items,
                'last_number': last_number,
                'expected_number': expected_number
            }
        
        # Function to check if continuation maintains structure coherence
        def check_structure_coherence(original_structure, continuation_text):
            """Check if continuation maintains the structure/plan from original response using semantic similarity."""
            # Handle both dict and list formats for structure
            if isinstance(original_structure, dict):
                structure_items = original_structure.get('items', [])
                numbered_items = original_structure.get('numbered_items', [])
                expected_number = original_structure.get('expected_number', 0)
            else:
                structure_items = original_structure if isinstance(original_structure, list) else []
                numbered_items = []
                expected_number = 0
            
            if not structure_items or len(structure_items) < 2:
                return True  # No clear structure to maintain
            
            continuation_lower = continuation_text.lower()
            
            # Check if continuation mentions any structure elements
            structure_mentions = sum(1 for struct_elem in structure_items if struct_elem in continuation_lower)
            
            # If continuation is substantial (>200 chars) but mentions no structure elements, might be off-plan
            if len(continuation_text) > 200 and structure_mentions == 0:
                # Use semantic similarity to check if continuation is still related to structure
                structure_text = ' '.join(structure_items)
                similarity = compute_semantic_similarity(structure_text, continuation_text)
                
                logger.debug(f"Structure coherence check: mentions={structure_mentions}, similarity={similarity:.3f}")
                
                # If similarity is very low (<0.2), likely off-plan (lowered for more graceful detection)
                if similarity < 0.2:
                    return False
            
            # Check for sequence continuation (e.g., if original had "1.", "2.", check for "3.")
            if numbered_items and expected_number > 0:
                # Check if continuation continues the numbering
                continuation_has_numbering = any(
                    continuation_lower.startswith(f"{i}.") or 
                    f"{i}." in continuation_lower[:50]
                    for i in range(1, 10)
                )
                # If original had numbering but continuation doesn't, might be off-plan
                # But only if continuation is substantial
                if not continuation_has_numbering and len(continuation_text) > 300:
                    # Check semantic similarity to see if it's still on-topic
                    structure_text = ' '.join(structure_items)
                    similarity = compute_semantic_similarity(structure_text, continuation_text)
                    logger.debug(f"Numbering coherence check: has_numbering={continuation_has_numbering}, similarity={similarity:.3f}")
                    if similarity < 0.25:
                        return False
            
            return True
        
        # Function to compute semantic similarity using embedding model
        def compute_semantic_similarity(text1, text2):
            """Compute semantic similarity between two texts using embedding model."""
            try:
                # Use global embedding model (loaded at startup)
                global global_embedding_model
                if global_embedding_model is None:
                    initialize_embedding_model()
                embed_model = global_embedding_model
                
                # Get embeddings
                emb1 = embed_model.get_text_embedding(text1[:1000])  # Limit to first 1000 chars
                emb2 = embed_model.get_text_embedding(text2[:1000])
                
                # Compute cosine similarity
                emb1 = np.array(emb1)
                emb2 = np.array(emb2)
                
                # Normalize
                norm1 = np.linalg.norm(emb1)
                norm2 = np.linalg.norm(emb2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                similarity = np.dot(emb1, emb2) / (norm1 * norm2)
                return float(similarity)
            except Exception as e:
                logger.warning(f"Error computing semantic similarity: {e}")
                return 0.0  # Return low similarity on error
        
        # Function to detect if continuation is off-topic or hallucinating
        def is_continuation_off_topic(original_text, continuation_text, original_message):
            """Detect if continuation text is off-topic or hallucinating using semantic similarity."""
            if not continuation_text or len(continuation_text.strip()) < 20:
                return False
            
            continuation_lower = continuation_text.lower()
            original_lower = original_text.lower()
            message_lower = original_message.lower()
            
            # Filter out model's internal reasoning tokens before checking
            continuation_clean = continuation_text
            internal_tokens = [
                '<end_of_instructions>', 'thought process:', 'thinking:',
                'reasoning:', 'internal:', 'meta:', '[thinking]', '[reasoning]',
                'end_of_instructions', 'thought process', 'thinking process'
            ]
            for token in internal_tokens:
                continuation_clean = continuation_clean.replace(token, '')
                continuation_clean = continuation_clean.replace(token.lower(), '')
            
            continuation_lower = continuation_clean.lower()
            
            # Red flags for hallucination/off-topic (only obvious ones):
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
            ]
            
            # Check for obvious hallucination indicators
            has_hallucination_indicators = any(indicator in continuation_lower for indicator in hallucination_indicators)
            if has_hallucination_indicators:
                return True
            
            # Use semantic similarity for more accurate topic coherence checking
            # Check similarity between continuation and original message
            message_similarity = compute_semantic_similarity(original_message, continuation_clean)
            
            # Check similarity between continuation and original response
            # Use last portion of original response for better context
            original_tail = original_text[-500:] if len(original_text) > 500 else original_text
            response_similarity = compute_semantic_similarity(original_tail, continuation_clean)
            
            # Log similarity scores for debugging
            logger.debug(f"Semantic similarity check: message={message_similarity:.3f}, response={response_similarity:.3f}")
            
            # If continuation is semantically very different from both message and response, likely off-topic
            # Threshold: similarity < 0.2 indicates low relevance (lowered for more graceful detection)
            if message_similarity < 0.2 and response_similarity < 0.2 and len(continuation_clean) > 150:
                logger.warning(f"Low semantic similarity detected: message={message_similarity:.3f}, response={response_similarity:.3f}")
                return True
            
            # Extract key medical/context words from original message and response
            medical_keywords = [
                'medical', 'health', 'treatment', 'patient', 'symptom', 'diagnosis',
                'medication', 'therapy', 'clinical', 'disease', 'condition', 'disorder',
                'migraine', 'headache', 'pain', 'chronic', 'acute', 'preventive',
                'pharmacological', 'dosage', 'side effect', 'contraindication', 'drug',
                'prescription', 'dosage', 'mg', 'tablet', 'capsule', 'injection',
                'preventive', 'prophylactic', 'abortive', 'rescue', 'maintenance'
            ]
            
            # Check if continuation contains medical keywords from original context
            continuation_has_medical = any(keyword in continuation_lower for keyword in medical_keywords)
            original_has_medical = any(keyword in original_lower for keyword in medical_keywords)
            
            # If original had medical context but continuation doesn't AND low similarity, likely off-topic
            if original_has_medical and not continuation_has_medical and len(continuation_text) > 150:
                if message_similarity < 0.25:  # Lower threshold for medical context (more graceful)
                    # Check if continuation is about completely different topics
                    non_medical_topics = ['fashion', 'anime', 'manga', 'culture', 'internet', 'viral', 'trend', 'kawaii']
                    if any(topic in continuation_lower for topic in non_medical_topics):
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
        
        # Function to check if continuation is completing the answer plan using semantic similarity
        def is_continuation_on_plan(original_text, continuation_text, original_message):
            """Check if continuation is actually continuing the original answer plan using semantic similarity.
            
            This uses embedding-based semantic checking for more accurate plan coherence.
            """
            if not continuation_text or len(continuation_text.strip()) < 20:
                return True  # Too short to judge, assume OK
            
            # Filter out model's internal reasoning tokens
            continuation_clean = continuation_text
            internal_tokens = [
                '<end_of_instructions>', 'thought process:', 'thinking:',
                'reasoning:', 'internal:', 'meta:', '[thinking]', '[reasoning]'
            ]
            for token in internal_tokens:
                continuation_clean = continuation_clean.replace(token, '')
            
            continuation_lower = continuation_clean.lower()
            
            # Extract structure from original response
            original_structure = extract_answer_structure(original_text)
            
            # Handle both dict and list formats
            if isinstance(original_structure, dict):
                structure_items = original_structure.get('items', [])
            else:
                structure_items = original_structure if isinstance(original_structure, list) else []
            
            # Use semantic similarity to check plan coherence
            # Get the last portion of original response for context
            original_tail = original_text[-500:] if len(original_text) > 500 else original_text
            
            # Compute semantic similarity between continuation and original response tail
            response_similarity = compute_semantic_similarity(original_tail, continuation_clean)
            
            # Compute semantic similarity between continuation and original message
            message_similarity = compute_semantic_similarity(original_message, continuation_clean)
            
            # If continuation is semantically similar to original response (>0.5), likely on-plan
            if response_similarity > 0.5:
                # Also check structure coherence
                if structure_items and len(structure_items) > 2:
                    structure_coherent = check_structure_coherence(original_structure, continuation_clean)
                    if not structure_coherent and response_similarity < 0.5:
                        # Low similarity and structure incoherence = likely off-plan (lowered for more graceful detection)
                        return False
                return True
            
            # If similarity is low, check structure mentions
            if structure_items and len(structure_items) > 3:
                structure_mentions = sum(1 for struct_elem in structure_items if struct_elem in continuation_lower)
                # Only flag if no structure mentions AND continuation is long (>200 chars) AND low similarity (lowered for more graceful detection)
                if structure_mentions == 0 and len(continuation_text) > 200 and response_similarity < 0.3:
                    # Check if it's actually medical content (might be OK even without structure mention)
                    medical_keywords = [
                        'medical', 'health', 'treatment', 'patient', 'symptom', 'diagnosis',
                        'medication', 'therapy', 'clinical', 'disease', 'condition', 'disorder'
                    ]
                    has_medical = any(keyword in continuation_lower for keyword in medical_keywords)
                    if not has_medical:
                        # No structure AND no medical content AND low similarity - likely off-topic
                        return False
            
            # Check if continuation starts with obviously wrong topics (very strict check)
            first_100_chars = continuation_lower[:100]
            obviously_wrong = any(first_100_chars.startswith(wrong) for wrong in 
                                 ['kawaii', 'wikipedia:', 'japanisch', 'deutsch', 'anime', 'manga'])
            
            if obviously_wrong:
                return False
            
            # If similarity is very low (<0.2) for both message and response, likely off-plan (lowered for more graceful detection)
            if response_similarity < 0.2 and message_similarity < 0.2 and len(continuation_clean) > 200:
                return False
            
            # If we get here, assume it's on plan (less strict)
            return True
        
        # Check if response is complete
        is_complete = is_response_complete(final_text)
        
        # Log completeness check result for debugging
        if is_complete:
            logger.info(f"Response marked as complete: {len(final_text)} chars, ends with: '{final_text[-50:]}'")
        else:
            logger.info(f"Response marked as incomplete: {len(final_text)} chars, ends with: '{final_text[-50:]}'")
        
        # Function to detect if continuation is repeating previous content using semantic similarity
        def compute_token_overlap(text1, text2):
            """Compute token overlap percentage between two texts using word and n-gram overlap.
            
            Returns a tuple: (word_overlap_ratio, ngram_overlap_ratio, overall_overlap)
            """
            import re
            
            # Normalize and tokenize texts (remove punctuation, lowercase, split)
            def tokenize(text):
                # Remove punctuation and normalize
                text = re.sub(r'[^\w\s]', ' ', text.lower())
                # Split into words
                words = [w for w in text.split() if len(w) > 2]  # Only words > 2 chars
                return words
            
            words1 = set(tokenize(text1))
            words2 = set(tokenize(text2))
            
            if not words1 or not words2:
                return (0.0, 0.0, 0.0)
            
            # Word overlap (Jaccard similarity)
            word_intersection = words1.intersection(words2)
            word_union = words1.union(words2)
            word_overlap = len(word_intersection) / len(word_union) if word_union else 0.0
            
            # N-gram overlap (bigrams and trigrams)
            def get_ngrams(words, n):
                return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))
            
            words1_list = tokenize(text1)
            words2_list = tokenize(text2)
            
            if len(words1_list) >= 2 and len(words2_list) >= 2:
                bigrams1 = get_ngrams(words1_list, 2)
                bigrams2 = get_ngrams(words2_list, 2)
                bigram_intersection = bigrams1.intersection(bigrams2)
                bigram_union = bigrams1.union(bigrams2)
                bigram_overlap = len(bigram_intersection) / len(bigram_union) if bigram_union else 0.0
            else:
                bigram_overlap = 0.0
            
            if len(words1_list) >= 3 and len(words2_list) >= 3:
                trigrams1 = get_ngrams(words1_list, 3)
                trigrams2 = get_ngrams(words2_list, 3)
                trigram_intersection = trigrams1.intersection(trigrams2)
                trigram_union = trigrams1.union(trigrams2)
                trigram_overlap = len(trigram_intersection) / len(trigram_union) if trigram_union else 0.0
            else:
                trigram_overlap = 0.0
            
            # Overall n-gram overlap (average of bigram and trigram)
            ngram_overlap = (bigram_overlap + trigram_overlap) / 2 if (bigram_overlap > 0 or trigram_overlap > 0) else 0.0
            
            # Overall overlap (weighted average: 40% words, 60% n-grams)
            overall_overlap = (word_overlap * 0.4) + (ngram_overlap * 0.6)
            
            return (word_overlap, ngram_overlap, overall_overlap)
        
        def is_continuation_repeating(previous_text, continuation_text):
            """Detect if continuation is repeating previous content using semantic similarity AND token overlap.
            
            Only flags as repetition if BOTH semantic similarity > 0.9 AND token overlap > 0.9 (90%+).
            This ensures we only remove true duplicates, not just similar content on the same topic.
            """
            if not continuation_text or len(continuation_text.strip()) < 50:
                return False
            
            # Use semantic similarity for topic-level checking
            semantic_similarity = compute_semantic_similarity(previous_text, continuation_text)
            
            # Compute token overlap (word and n-gram overlap)
            word_overlap, ngram_overlap, overall_overlap = compute_token_overlap(previous_text, continuation_text)
            
            logger.info(f"Repetition check: semantic={semantic_similarity:.3f}, word_overlap={word_overlap:.3f}, ngram_overlap={ngram_overlap:.3f}, overall={overall_overlap:.3f}")
            
            # Only flag as repetition if BOTH semantic similarity > 0.9 AND token overlap > 0.9 (90%+)
            # This ensures we only remove true duplicates, not just similar content on the same topic
            if semantic_similarity > 0.9 and overall_overlap > 0.9:
                logger.warning(f"High repetition detected: semantic={semantic_similarity:.3f}, token_overlap={overall_overlap:.3f}")
                return True
            
            # Also check if semantic similarity is very high (>0.95) AND token overlap is high (>0.85)
            # This catches near-duplicates
            if semantic_similarity > 0.95 and overall_overlap > 0.85:
                logger.warning(f"Near-duplicate detected: semantic={semantic_similarity:.3f}, token_overlap={overall_overlap:.3f}")
                return True
            
            # Also check for exact/similar sentence repetition using token overlap
            continuation_lower = continuation_text.lower().strip()
            previous_lower = previous_text.lower().strip()
            
            # Check for repeated sentences (exact matches and high token overlap)
            continuation_sentences = [s.strip() for s in continuation_lower.split('.') if len(s.strip()) > 30]
            previous_sentences = [s.strip() for s in previous_lower.split('.') if len(s.strip()) > 30]
            
            # Count how many continuation sentences appear in previous text
            repeated_sentences = 0
            for cont_sent in continuation_sentences[:5]:  # Check first 5 sentences
                if len(cont_sent) > 40:  # Only check substantial sentences
                    # Check if this sentence appears in previous text (exact match)
                    if cont_sent in previous_lower:
                        repeated_sentences += 1
                    # Also check token overlap for similar sentences (90%+ overlap)
                    else:
                        # Check token overlap with each previous sentence
                        for prev_sent in previous_sentences[:10]:  # Check first 10 previous sentences
                            if len(prev_sent) > 40:
                                # Check both semantic similarity and token overlap
                                sent_similarity = compute_semantic_similarity(prev_sent, cont_sent)
                                _, _, sent_overlap = compute_token_overlap(prev_sent, cont_sent)
                                
                                # Only flag if BOTH semantic similarity > 0.9 AND token overlap > 0.9
                                if sent_similarity > 0.9 and sent_overlap > 0.9:
                                    repeated_sentences += 1
                                    break
            
            # If more than 50% of sentences are repeated (with 90%+ overlap), likely repeating
            if len(continuation_sentences) > 0:
                repetition_ratio = repeated_sentences / min(len(continuation_sentences), 5)
                if repetition_ratio > 0.5:  # Increased threshold to 50%
                    logger.warning(f"High sentence repetition ratio: {repetition_ratio:.2f}")
                    return True
            
            # Check for repeated long phrases (4+ word phrases) using token overlap
            continuation_words = continuation_lower.split()
            if len(continuation_words) > 30:
                repeated_phrases = 0
                # Check for repeated 4-word phrases
                for i in range(len(continuation_words) - 3):
                    phrase = ' '.join(continuation_words[i:i+4])
                    if len(phrase) > 20:
                        # Check exact match first
                        if phrase in previous_lower:
                            repeated_phrases += 1
                        # Also check token overlap for similar phrases
                        else:
                            # Extract similar phrases from previous text
                            prev_words = previous_lower.split()
                            for j in range(len(prev_words) - 3):
                                prev_phrase = ' '.join(prev_words[j:j+4])
                                if len(prev_phrase) > 20:
                                    # Check both semantic similarity and token overlap
                                    phrase_similarity = compute_semantic_similarity(prev_phrase, phrase)
                                    _, _, phrase_overlap = compute_token_overlap(prev_phrase, phrase)
                                    
                                    # Only flag if BOTH semantic similarity > 0.9 AND token overlap > 0.9
                                    if phrase_similarity > 0.9 and phrase_overlap > 0.9:
                                        repeated_phrases += 1
                                        break
                        if repeated_phrases > 5:  # Increased threshold to 5 repeated phrases
                            logger.warning(f"High phrase repetition: {repeated_phrases} phrases")
                            return True
            
            return False
        
        # Sequential generation: continue generating if response is incomplete
        # This allows the model to go beyond max_new_tokens to complete its answer
        max_total_tokens = int(max_new_tokens * 5)  # Allow up to 5x max_new_tokens total (increased)
        continuation_chunk_size = int(max_new_tokens * 0.5)  # Generate 50% more tokens each continuation
        total_tokens_generated = chunk_count  # Approximate from chunks
        # Initialize continuation_count - use cached value if resuming, otherwise start at 0
        if not skip_initial_generation:
            continuation_count = 0
        # continuation_count already set from cache if resuming
        max_continuations = 10  # Increased to allow more continuations if on-topic
        original_response_before_continuation = final_text  # Store original response
        previous_continuation_texts = []  # Track previous continuations to detect repetition
        
        # Continue as long as response is incomplete, on-topic, not repeating, and within limits
        while continuation_count < max_continuations and total_tokens_generated < max_total_tokens:
            # Check if we should continue (incomplete and not repeating)
            if is_complete:
                logger.info(f"Response is complete - stopping continuation")
                break
            
            continuation_count += 1
            logger.info(f"Response incomplete - continuing generation (continuation {continuation_count}/{max_continuations})")
            logger.info(f"Current response length: {len(final_text)} chars, ends with: '{final_text[-100:]}'")
            
            # Add loader indicator to show sequential generation is happening
            loader_indicator = "\n\n*[Generating continuation answer...]*"
            updated_history[-1]["content"] = final_text + loader_indicator
            yield updated_history
            
            # Prepare continuation prompt more carefully to maintain context and answer plan
            # Extract the answer structure/plan from original response
            answer_structure = extract_answer_structure(original_response_before_continuation)
            
            # Log structure information for tracking
            if isinstance(answer_structure, dict):
                structure_items = answer_structure.get('items', [])
                numbered_items = answer_structure.get('numbered_items', [])
                expected_number = answer_structure.get('expected_number', 0)
                logger.info(f"Answer structure: {len(structure_items)} items, last number: {answer_structure.get('last_number', 0)}, expected: {expected_number}")
            else:
                structure_items = answer_structure if isinstance(answer_structure, list) else []
                numbered_items = []
                expected_number = 0
                logger.info(f"Answer structure: {len(structure_items)} items")
            
            # Use the last portion of the response to maintain context
            # But also include key structure elements to maintain the plan
            response_tail = final_text[-400:] if len(final_text) > 400 else final_text  # Last 400 chars
            
            # Build continuation prompt with explicit instructions to maintain plan
            structure_hint = ""
            if answer_structure:
                # Handle both dict and list formats
                if isinstance(answer_structure, dict):
                    structure_items = answer_structure.get('items', [])
                    numbered_items = answer_structure.get('numbered_items', [])
                    expected_number = answer_structure.get('expected_number', 0)
                else:
                    structure_items = answer_structure if isinstance(answer_structure, list) else []
                    numbered_items = []
                    expected_number = 0
                
                if structure_items:
                    # Include structure elements that haven't been completed yet
                    # Check which structure elements are already mentioned in the response
                    mentioned_elements = [elem for elem in structure_items if elem in final_text.lower()]
                    unmentioned_elements = [elem for elem in structure_items if elem not in mentioned_elements]
                    
                    # If there are unmentioned elements, include them in the hint
                    if unmentioned_elements:
                        structure_hint = f"\n[Answer structure/plan to continue: {', '.join(unmentioned_elements[:3])}...]"
                    else:
                        # All elements mentioned, check if numbering should continue
                        if numbered_items and expected_number > 0:
                            # Continue numbering sequence
                            structure_hint = f"\n[Continue from item {expected_number} in the answer plan. Maintain the same structure and numbering sequence.]"
                        else:
                            # Just continue the sequence
                            structure_hint = f"\n[Answer structure/plan: {', '.join(structure_items[-2:])}...]"
            
            # Build continuation prompt - methodical and direct to maintain content flow
            # Track what has been covered to guide continuation
            covered_topics = []
            if structure_items:
                # Check which structure items are already mentioned
                for item in structure_items:
                    if item.lower() in final_text.lower():
                        covered_topics.append(item)
            
            # Build methodical continuation instruction
            continuation_instruction = "Continue the medical response from where it left off. "
            if covered_topics:
                continuation_instruction += f"Topics already covered: {', '.join(covered_topics[:3])}. "
            if structure_hint:
                continuation_instruction += f"{structure_hint.strip()} "
            continuation_instruction += "Continue naturally from the last sentence. Do not repeat information already provided. Do not include meta-commentary or instructions."
            
            # Include semantic context hint using embedding similarity
            continuation_prompt = (
                f"{prompt}\n\n"
                f"[Previous response (incomplete):]\n{response_tail}\n\n"
                f"{continuation_instruction}"
            )
            
            logger.info(f"Continuation prompt length: {len(continuation_prompt)} chars, structure hint: {bool(structure_hint)}")
            
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
                    
                    # Check for hallucination after collecting some text (less frequent checks)
                    # Only check every 30 chunks to avoid false positives (increased from 20)
                    if len(continuation_text) > 150 and continuation_chunk_count % 30 == 0:
                        # Filter internal tokens before checking
                        check_text = continuation_text
                        internal_check_tokens = [
                            '<end_of_instructions>', '<end_of_turn>', 'thought process:', 
                            'thinking:', 'reasoning:', 'the prompt asks you', 
                            'the user has stopped', 'provide steps involved'
                        ]
                        for token in internal_check_tokens:
                            check_text = check_text.replace(token, '').replace(token.lower(), '')
                            check_text = check_text.replace(token.upper(), '')
                        
                        # Check for off-topic/hallucination (only obvious ones)
                        if is_continuation_off_topic(original_response_before_continuation, check_text, message):
                            logger.warning(f"Hallucination/off-topic detected in continuation {continuation_count} after {len(continuation_text)} chars")
                            logger.warning(f"Off-topic text sample: '{check_text[:200]}...'")
                            off_topic_detected = True
                            # Stop the continuation thread
                            stop_event.set()
                            break
                        
                        # Only log plan deviation warnings, don't stop (less strict)
                        # Check plan coherence less frequently to avoid false positives
                        if continuation_chunk_count % 30 == 0:
                            if not is_continuation_on_plan(original_response_before_continuation, check_text, message):
                                # Only log, don't stop - continuation might still be valid
                                logger.debug(f"Continuation {continuation_count} may not be following the answer plan after {len(continuation_text)} chars")
                    
                    # Update UI with continuation (remove loader indicator once we have content)
                    # Remove loader indicator on first chunk
                    if continuation_chunk_count == 1:
                        # Remove loader indicator from partial_response if present
                        partial_response_clean = partial_response.replace("*[Generating continuation...]*", "").strip()
                        final_text = (partial_response_clean + continuation_text).strip()
                    else:
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
            
            # Filter out model's internal reasoning tokens from continuation before appending
            continuation_clean = continuation_text
            internal_tokens = [
                '<end_of_instructions>', '<end_of_turn>', 'thought process:', 'thinking:',
                'reasoning:', 'internal:', 'meta:', '[thinking]', '[reasoning]',
                'end_of_instructions', 'end_of_turn', 'thought process', 'thinking process',
                'okay, i need to finish', 'i need to finish', 'as though someone else',
                '<end_of_turn>thought', '<end_of_turn>', 'end_of_turn', 'thought',
                'the prompt asks you', 'the user has stopped', 'provide steps involved',
                'so provide steps', 'steps involved in developing'
            ]
            for token in internal_tokens:
                continuation_clean = continuation_clean.replace(token, '')
                continuation_clean = continuation_clean.replace(token.lower(), '')
                continuation_clean = continuation_clean.replace(token.upper(), '')
            
            # Remove any lines that start with these patterns
            lines = continuation_clean.split('\n')
            filtered_lines = []
            for line in lines:
                line_lower = line.lower().strip()
                # Skip lines that are clearly internal reasoning
                if any(line_lower.startswith(pattern) for pattern in [
                    'thought process', 'thinking:', 'reasoning:', 'internal:',
                    'okay, i need', 'i need to finish', 'as though someone',
                    'the prompt asks you', 'the user has stopped', 'provide steps',
                    'so provide steps', 'steps involved'
                ]) or any(pattern in line_lower for pattern in [
                    '<end_of_turn>', 'end_of_turn', 'thought process',
                    'the prompt asks', 'the user has stopped responding'
                ]):
                    continue
                filtered_lines.append(line)
            continuation_clean = '\n'.join(filtered_lines).strip()
            
            # Deduplication: Remove duplicate content using semantic similarity
            # Check if continuation is repeating previous content
            if continuation_clean and len(continuation_clean) > 50:
                # Check against original response
                if is_continuation_repeating(original_response_before_continuation, continuation_clean):
                    logger.warning(f"Continuation {continuation_count} is repeating original response - removing duplicates")
                    # Try to extract only new content by comparing sentences
                    continuation_sentences = [s.strip() for s in continuation_clean.split('.') if len(s.strip()) > 30]
                    original_sentences = [s.strip() for s in original_response_before_continuation.split('.') if len(s.strip()) > 30]
                    
                    new_sentences = []
                    for cont_sent in continuation_sentences:
                        # Check if this sentence is new (not in original)
                        if cont_sent.lower() not in original_response_before_continuation.lower():
                            # Also check BOTH semantic similarity AND token overlap (90%+ for both)
                            is_new = True
                            for orig_sent in original_sentences[-10:]:  # Check last 10 sentences
                                sent_similarity = compute_semantic_similarity(orig_sent, cont_sent)
                                _, _, sent_overlap = compute_token_overlap(orig_sent, cont_sent)
                                
                                # Only flag as duplicate if BOTH semantic similarity > 0.9 AND token overlap > 0.9
                                if sent_similarity > 0.9 and sent_overlap > 0.9:
                                    is_new = False
                                    logger.debug(f"Duplicate sentence detected: semantic={sent_similarity:.3f}, overlap={sent_overlap:.3f}")
                                    break
                            if is_new:
                                new_sentences.append(cont_sent)
                    
                    if new_sentences:
                        continuation_clean = '. '.join(new_sentences) + '.'
                        logger.info(f"Extracted {len(new_sentences)} new sentences from continuation")
                    else:
                        logger.warning(f"Continuation {continuation_count} contains only duplicate content - discarding")
                        continuation_clean = ""
                
                # Also check against previous continuations
                if continuation_clean and previous_continuation_texts:
                    for prev_cont in previous_continuation_texts[-2:]:  # Check last 2 continuations
                        if is_continuation_repeating(prev_cont, continuation_clean):
                            logger.warning(f"Continuation {continuation_count} is repeating previous continuation - removing duplicates")
                            # Extract only new sentences
                            continuation_sentences = [s.strip() for s in continuation_clean.split('.') if len(s.strip()) > 30]
                            prev_sentences = [s.strip() for s in prev_cont.split('.') if len(s.strip()) > 30]
                            
                            new_sentences = []
                            for cont_sent in continuation_sentences:
                                if cont_sent.lower() not in prev_cont.lower():
                                    is_new = True
                                    for prev_sent in prev_sentences[-10:]:
                                        # Check BOTH semantic similarity AND token overlap (90%+ for both)
                                        sent_similarity = compute_semantic_similarity(prev_sent, cont_sent)
                                        _, _, sent_overlap = compute_token_overlap(prev_sent, cont_sent)
                                        
                                        # Only flag as duplicate if BOTH semantic similarity > 0.9 AND token overlap > 0.9
                                        if sent_similarity > 0.9 and sent_overlap > 0.9:
                                            is_new = False
                                            logger.debug(f"Duplicate sentence detected: semantic={sent_similarity:.3f}, overlap={sent_overlap:.3f}")
                                            break
                                    if is_new:
                                        new_sentences.append(cont_sent)
                            
                            if new_sentences:
                                continuation_clean = '. '.join(new_sentences) + '.'
                            else:
                                continuation_clean = ""
                                break
            
            # Append cleaned continuation to response only if it's on-topic and not empty
            if continuation_clean and len(continuation_clean.strip()) > 20:
                # Log continuation content for tracking
                logger.info(f"Continuation {continuation_count} added: {len(continuation_clean)} chars, starts with: '{continuation_clean[:100]}'")
                partial_response += continuation_clean
                final_text = partial_response.strip()
                
                # Save cache after each continuation
                if cache_key and final_text:
                    _save_cache(cache_key, final_text, continuation_count, {
                        'chunk_count': chunk_count,
                        'continuation_chunk_count': continuation_chunk_count,
                        'stage': 'continuation'
                    })
                
                # Track continuation content methodically
                continuation_summary = {
                    'count': continuation_count,
                    'length': len(continuation_clean),
                    'start': continuation_clean[:50],
                    'end': continuation_clean[-50:] if len(continuation_clean) > 50 else continuation_clean
                }
                logger.info(f"Continuation summary: {continuation_summary}")
            else:
                # If all continuation was filtered out or is duplicate, might be all internal reasoning
                logger.warning(f"Continuation {continuation_count} was mostly internal reasoning or duplicate - discarding")
                logger.warning(f"Filtered continuation length: {len(continuation_clean) if continuation_clean else 0} chars")
                final_text = original_response_before_continuation
                partial_response = original_response_before_continuation
                is_complete = True
                break
            chunk_count += continuation_chunk_count
            total_tokens_generated += continuation_chunk_count
            
            # Final checks for hallucination and plan coherence on the cleaned continuation
            # Use cleaned continuation for checks
            if is_continuation_off_topic(original_response_before_continuation, continuation_clean, message):
                logger.warning(f"Hallucination detected in full continuation {continuation_count} - discarding")
                # Revert to original response
                final_text = original_response_before_continuation
                partial_response = original_response_before_continuation
                is_complete = True
                break
            
            # Check if continuation is following the plan (less strict - only flag obvious issues)
            # Only check if cleaned continuation is substantial (>100 chars)
            if len(continuation_clean) > 100:
                if not is_continuation_on_plan(original_response_before_continuation, continuation_clean, message):
                    # Check if it's actually off-topic (not just plan deviation)
                    if is_continuation_off_topic(original_response_before_continuation, continuation_clean, message):
                        # If it's both off-plan AND off-topic, discard it
                        logger.warning(f"Discarding continuation {continuation_count} - off-plan and off-topic")
                        final_text = original_response_before_continuation
                        partial_response = original_response_before_continuation
                        is_complete = True
                        break
                    else:
                        # If it's off-plan but still on-topic, accept it (might be natural continuation)
                        logger.info(f"Accepting continuation {continuation_count} - off-plan but on-topic")
            
            # Check for repetition - if continuation is repeating previous content, stop
            is_repeating = False
            if previous_continuation_texts:
                # Check if this continuation repeats any previous continuation
                for prev_cont in previous_continuation_texts[-2:]:  # Check last 2 continuations
                    if is_continuation_repeating(prev_cont, continuation_clean):
                        logger.warning(f"Continuation {continuation_count} is repeating previous content - stopping")
                        is_repeating = True
                        break
            
            # Also check if continuation repeats the original response
            if not is_repeating and len(continuation_clean) > 100:
                if is_continuation_repeating(original_response_before_continuation, continuation_clean):
                    logger.warning(f"Continuation {continuation_count} is repeating original response - stopping")
                    is_repeating = True
            
            if is_repeating:
                # Revert to state before this continuation
                final_text = partial_response  # partial_response already has previous continuations
                is_complete = True  # Mark as complete to stop
                break
            
            # Store this continuation for repetition checking
            previous_continuation_texts.append(continuation_clean)
            
            # Check if continuation is complete
            is_complete = is_response_complete(final_text)
            
            # Log completeness check result for debugging
            if is_complete:
                logger.info(f"Continuation {continuation_count} marked as complete: {len(final_text)} chars, ends with: '{final_text[-50:]}'")
            else:
                logger.info(f"Continuation {continuation_count} marked as incomplete: {len(final_text)} chars, ends with: '{final_text[-50:]}'")
            
            logger.info(f"Continuation {continuation_count} complete: {len(continuation_clean)} chars (cleaned from {len(continuation_text)}), {continuation_chunk_count} chunks. Total: {len(final_text)} chars. Complete: {is_complete}")
            
            # If we got very few tokens in continuation, might be done
            if continuation_chunk_count < 10:
                logger.info("Continuation produced very few tokens, assuming complete")
                is_complete = True
            
            # Check if we have enough information (response is substantial)
            # If response is very long (>5000 chars) and seems complete enough, mark as complete
            if len(final_text) > 5000 and not is_complete:
                # Check if last 200 chars suggest completion
                last_200 = final_text[-200:].lower()
                completion_indicators = [
                    'conclusion', 'summary', 'in summary', 'to summarize',
                    'important note', 'remember', 'always consult',
                    'consult your', 'see your doctor', 'seek medical'
                ]
                if any(indicator in last_200 for indicator in completion_indicators):
                    logger.info(f"Response is substantial ({len(final_text)} chars) and has completion indicators - marking as complete")
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
        
        # Save final cache before yielding
        if cache_key and final_text:
            _save_cache(cache_key, final_text, continuation_count, {
                'chunk_count': chunk_count,
                'stage': 'final',
                'is_complete': is_complete
            })
        
        updated_history[-1]["content"] = final_text
        yield updated_history

    except GeneratorExit:
        stop_event.set()
        thread.join()
        # Save cache on exit for recovery
        if cache_key and partial_response:
            _save_cache(cache_key, partial_response, continuation_count, {'stage': 'interrupted'})
        raise
    except Exception as e:
        logger.exception(f"streaming error: {e}")
        stop_event.set()
        # Save cache on error for recovery
        if cache_key and partial_response:
            _save_cache(cache_key, partial_response, continuation_count, {'stage': 'error', 'error': str(e)[:200]})
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
    initialize_embedding_model()
    demo = create_demo()
    demo.launch()