---
title: MedSwin Distilled 7B
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
- inference-api
license: apache-2.0
short_description: MedAlpaca-7B SFT & MedGemma-27b KD with RAG
---

### Medical RAG

- **LLM**: `MedAI-COS30018/MedSwin-7B-Distilled` (bf16) for response generation using `transformers`.
- **Embeddings**: `abhinand/MedEmbed-large-v0.1` via `llama_index.embeddings.huggingface` for dense retrieval.
- **Indexing**:
  - Upload `.pdf`/`.txt` → extract text → hierarchical chunking (2048/512/128, 20 overlap).
  - Store nodes with `SimpleDocumentStore`; build `VectorStoreIndex`; persisted per-session.
- **Retrieval**: top-k similarity + `AutoMergingRetriever` (merge threshold configurable) to produce final context.
- **Prompting**: chat template from tokenizer; system prompt emphasizes evidence-based, non-diagnostic guidance.
