# 🍲 Zomato RAG Chatbot — Conversational Discovery Engine

A Refactored Retrieval-Augmented Generation (RAG) pipeline designed to act as an intuitive, warm, and conversational food and restaurant discovery assistant for Zomato. Built on **LlamaIndex**, **ChromaDB**, and powered by **Groq (`llama-3.1-8b-instant`)** and **HuggingFace (`BAAI/bge-small-en-v1.5`)**.

---

## 🌟 Key Features

* **Narrative Document Builder**: Converts tabular CSV data, raw review lists, and NLP sentiment metrics into coherent, human-readable prose before vector embedding to maximize retrieval quality.
* **Refactored System Instructions**: Custom system prompt strictly forbids raw metadata dumps (e.g., `'sentiment_score'`, `'positive_ratio'`) or token array regurgitation in favor of natural summaries.
* **Temperature Tuning**: Configured to `0.4` for an optimal balance between strict factual precision and warm conversational engagement.
* **Persistent ChromaDB Vector Store**: Indexes restaurant documents and FAQs using chunk-aware vector embeddings (`BAAI/bge-small-en-v1.5`) with persistent local storage.
* **Interactive CLI Chat Engine**: Employs LlamaIndex's `condense_plus_context` chat mode with real-time query support and a `sources` command to inspect retrieved context nodes.
* **Automated RAG Evaluation**: Built-in test suite evaluating **Faithfulness** and **Relevancy** metrics via LlamaIndex evaluators, exporting results directly to CSV.

---

## 🛠️ Architecture & Data Pipeline
