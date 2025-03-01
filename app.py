import gradio as gr
import os
import PyPDF2
import logging
import torch
import threading
import time
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

MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment variables")

# Custom UI
TITLE = "<h1><center>Multi-Document RAG with LLama 3.1-8B Model</center></h1>"
DESCRIPTION = """
<center>
<p>Upload PDF or text files to get started!</p>
<p>After asking question wait for RAG system to get relevant nodes and pass to LLM</p>
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
    background: #34c759 !important;
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
    background: #1a73e8 !important;
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

def initialize_model_and_tokenizer():
    global global_model, global_tokenizer
    if global_model is None or global_tokenizer is None:
        logger.info("Initializing model and tokenizer...")
        global_tokenizer = AutoTokenizer.from_pretrained(MODEL, token=HF_TOKEN)
        global_model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN,
            torch_dtype=torch.float16
        )
        logger.info("Model and tokenizer initialized successfully")

def get_llm(temperature=0.7, max_new_tokens=256, top_p=0.95, top_k=50):
    global global_model, global_tokenizer
    if global_model is None or global_tokenizer is None:
        initialize_model_and_tokenizer()
    
    return HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=max_new_tokens,
        tokenizer=global_tokenizer,
        model=global_model,
        generate_kwargs={
            "do_sample": True,
            "temperature": temperature,
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
    llm = get_llm()
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
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    penalty: float,
    retriever_k: int,
    merge_threshold: float,
    request: gr.Request
):
    if not request:
        yield history + [{"role": "assistant", "content": "Session initialization failed. Please refresh the page."}]
        return
    user_id = request.session_hash
    index_dir = f"./{user_id}_index"
    if not os.path.exists(index_dir):
        yield history + [{"role": "assistant", "content": "Please upload documents first."}]
        return

    max_new_tokens = int(max_new_tokens) if isinstance(max_new_tokens, (int, float)) else 1024
    temperature = float(temperature) if isinstance(temperature, (int, float)) else 0.9  
    top_p = float(top_p) if isinstance(top_p, (int, float)) else 0.95  
    top_k = int(top_k) if isinstance(top_k, (int, float)) else 50  
    penalty = float(penalty) if isinstance(penalty, (int, float)) else 1.2
    retriever_k = int(retriever_k) if isinstance(retriever_k, (int, float)) else 15
    merge_threshold = float(merge_threshold) if isinstance(merge_threshold, (int, float)) else 0.5
    llm = get_llm(temperature=temperature, max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k)
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL, token=HF_TOKEN)
    Settings.llm = llm
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
    logger.info(f"Query: {message}")
    retrieval_start = time.time()
    base_nodes = base_retriever.retrieve(message)
    logger.info(f"Retrieved {len(base_nodes)} base nodes in {time.time() - retrieval_start:.2f}s")
    base_file_sources = {}
    for node in base_nodes:
        if hasattr(node.node, 'metadata') and 'file_name' in node.node.metadata:
            file_name = node.node.metadata['file_name']
            if file_name not in base_file_sources:
                base_file_sources[file_name] = 0
            base_file_sources[file_name] += 1
    logger.info(f"Base retrieval file distribution: {base_file_sources}")
    merging_start = time.time()
    merged_nodes = auto_merging_retriever.retrieve(message)
    logger.info(f"Retrieved {len(merged_nodes)} merged nodes in {time.time() - merging_start:.2f}s")
    merged_file_sources = {}
    for node in merged_nodes:
        if hasattr(node.node, 'metadata') and 'file_name' in node.node.metadata:
            file_name = node.node.metadata['file_name']
            if file_name not in merged_file_sources:
                merged_file_sources[file_name] = 0
            merged_file_sources[file_name] += 1
    logger.info(f"Merged retrieval file distribution: {merged_file_sources}")
    context = "\n\n".join([n.node.text for n in merged_nodes])
    source_info = ""
    if merged_file_sources:
        source_info = "\n\nRetrieved information from files: " + ", ".join(merged_file_sources.keys())
    formatted_system_prompt = f"{system_prompt}\n\nDocument Context:\n{context}{source_info}"
    messages = [{"role": "system", "content": formatted_system_prompt}]
    for entry in history:
        messages.append(entry)
    messages.append({"role": "user", "content": message})
    prompt = global_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    stop_event = threading.Event()
    class StopOnEvent(StoppingCriteria):
        def __init__(self, stop_event):
            super().__init__()
            self.stop_event = stop_event

        def __call__(self, input_ids, scores, **kwargs):
            return self.stop_event.is_set()
    stopping_criteria = StoppingCriteriaList([StopOnEvent(stop_event)])
    streamer = TextIteratorStreamer(
        global_tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    inputs = global_tokenizer(prompt, return_tensors="pt").to(global_model.device)
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=penalty,
        do_sample=True,
        stopping_criteria=stopping_criteria
    )
    thread = threading.Thread(target=global_model.generate, kwargs=generation_kwargs)
    thread.start()
    updated_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""}
    ]
    yield updated_history
    partial_response = ""
    try:
        for new_text in streamer:
            partial_response += new_text
            updated_history[-1]["content"] = partial_response
            yield updated_history
        output_ids = global_tokenizer.encode(partial_response, return_tensors="pt")
        yield updated_history
    except GeneratorExit:
        stop_event.set()
        thread.join()
        raise

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
                        value="As a knowledgeable assistant, your task is to provide detailed and context-rich answers based on the relevant information from all uploaded documents. When information is sourced from multiple documents, summarize the key points from each and explain how they relate, noting any connections or contradictions. Your response should be thorough, informative, and easy to understand.",
                        label="System Prompt",
                        lines=3
                    )
                    
                    with gr.Tab("Generation Parameters"):
                        temperature = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=0.9,  
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
                            value=0.95, 
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
                            value=1.2,
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

                submit_button.click(
                    fn=stream_chat,
                    inputs=[
                        message_input, 
                        chatbot, 
                        system_prompt, 
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