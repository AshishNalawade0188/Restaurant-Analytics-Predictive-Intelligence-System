```markdown
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


```

┌───────────────────────────┐
│ Enriched Zomato CSV Data  │
└─────────────┬─────────────┘
│
▼
┌───────────────────────────┐
│ Narrative Document Builder│  <-- Synthesizes stats, menus & reviews into prose
└─────────────┬─────────────┘
│
▼
┌───────────────────────────┐
│ SentenceSplitter Chunking │  <-- 400 tokens / 60 overlap
└─────────────┬─────────────┘
│
▼
┌───────────────────────────┐
│   ChromaDB Storage Path   │  <-- BAAI/bge-small-en-v1.5 Embeddings
└─────────────┬─────────────┘
│
▼
┌───────────────────────────┐
│  Condense + Context Chat  │  <-- Powered by Llama-3.1-8B-Instant via Groq
└───────────────────────────┘

```

---

## 📋 Prerequisites

* **Python**: `3.9+`
* **Groq API Key**: Obtain a free API key from [Groq Cloud](https://console.groq.com/).

---

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/your-username/zomato-rag-chatbot.git](https://github.com/your-username/zomato-rag-chatbot.git)
   cd zomato-rag-chatbot

```

2. **Install required dependencies**:
```bash
pip install pandas chromadb python-dotenv llama-index llama-index-embeddings-huggingface llama-index-llms-openai-like llama-index-vector-stores-chroma

```


3. **Set up Environment Variables**:
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here

```



---

## ⚙️ Configuration

Inside the main python script, edit the following constants under the `CONFIG` section to match your local setup:

| Variable | Default Value | Description |
| --- | --- | --- |
| `CSV_PATH` | `r"C:\...\restaurant_reviews_enriched_imputed.csv"` | Path to your preprocessed Zomato CSV dataset. |
| `SAMPLE_SIZE` | `1000` | Number of rows to sample (`None` for full dataset). |
| `REBUILD_INDEX` | `False` | Set to `True` to wipe and recreate the ChromaDB collection. |
| `CHROMA_PATH` | `"./chroma_storage"` | Directory for persistent ChromaDB storage. |
| `COLLECTION_NAME` | `"zomato_rag"` | Name of the ChromaDB collection. |
| `CHUNK_SIZE` | `400` | Chunk size in tokens for text node splitting. |
| `CHUNK_OVERLAP` | `60` | Token overlap between consecutive chunks. |

---

## 🚀 Usage

### 1. Launch Interactive Chatbot

Run the script directly to start the interactive chat CLI:

```bash
python main.py

```

**Chat Commands:**

* `exit` / `quit`: Terminate the session.
* `sources`: View score and metadata for the context nodes retrieved in the last response.

*Example Output:*

```text
Zomato RAG chatbot ready. Type 'exit' to quit, 'sources' to see last retrieval.

You: Recommend a good North Indian place in Koramangala.
Bot: You might enjoy Empire Restaurant located in Koramangala. They are well-known for North Indian and Biryani dishes with an average cost of ₹700 for two...

You: sources
  - [restaurant] Empire Restaurant (score=0.842)

```

### 2. Run Response Evaluation

Evaluate response quality against predefined benchmark questions using LlamaIndex `FaithfulnessEvaluator` and `RelevancyEvaluator`:

```bash
python main.py --evaluate

```

This generates an `evaluation_report.csv` containing passing states and detailed feedback metrics.

### 3. Build Vector Index Only (No Chat)

Build or rebuild the ChromaDB store without triggering the interactive conversation:

```bash
python main.py --no-chat

```

---

## 📊 Evaluation Metrics

The script runs automated evaluation using LlamaIndex core evaluators across two core metrics:

1. **Faithfulness**: Verifies whether the LLM's response stays strictly truthful to the retrieved context without hallucinating details.
2. **Relevancy**: Measures if the generated output directly addresses the user's specific query.

---

## 📄 License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).

```

```
