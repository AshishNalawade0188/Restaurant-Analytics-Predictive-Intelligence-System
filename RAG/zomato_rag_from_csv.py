"""
Zomato RAG Chatbot — CSV-driven Pipeline
==========================================
Loads a real Zomato CSV (your cleaned + logically-imputed version), builds
per-restaurant + FAQ documents, indexes into ChromaDB, and runs an
interactive chat loop.

Fixes vs. the earlier prototype:
  - Idempotent indexing: REBUILD_INDEX wipes the Chroma collection before
    rebuilding, so re-running the script doesn't duplicate documents.
  - NaN-safe metadata: sanitize_metadata() converts NaN/NaT to None before
    anything touches Chroma (some vector store backends reject raw NaN).
  - COLUMN_MAP: your cleaned CSV likely renamed columns during preprocessing.
    Edit this ONE dict instead of hunting through the code.
"""

import sys
import math
import os
from ast import literal_eval

import pandas as pd
import chromadb
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

# ---------------------------------------------------------------------------
# CONFIG — edit these for your setup
# ---------------------------------------------------------------------------
CSV_PATH = r"C:\Users\admin\Desktop\Zomato_ML_Notes\RAG\zomato_rag_cleaned.csv"   # <-- point this at your real cleaned CSV
SAMPLE_SIZE = 1000          # e.g. 1000 to test on a subset first; None = full dataset
REBUILD_INDEX = False       # True = re-embed everything (only when CSV/doc logic changed).
                             # False = reuse existing Chroma collection, skip embedding entirely.
CHROMA_PATH = "./chroma_storage"
COLLECTION_NAME = "zomato_rag"

# If your cleaned CSV renamed columns during preprocessing, edit the VALUES
# here (left = what this script expects, right = your actual column name).
COLUMN_MAP = {
    "name": "name",
    "location": "location",
    "rest_type": "rest_type",
    "cuisines": "cuisines",
    "dish_liked": "dish_liked",
    "cost_for_two": "approx_cost(for two people)",
    "rate": "rate",
    "votes": "votes",
    "online_order": "online_order",
    "book_table": "book_table",
    "menu_item": "menu_item",
    "reviews_list": "reviews_list",
    "listed_in_type": "listed_in(type)",
}

# ---------------------------------------------------------------------------
# STEP 0: Model settings — swap for your real setup (see prior script's notes)
# ---------------------------------------------------------------------------
USE_REAL_MODELS = True

if USE_REAL_MODELS:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.openai_like import OpenAILike

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. In PowerShell, run: "
            '$env:GROQ_API_KEY = "your-groq-key"'
        )

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = OpenAILike(
        model="llama-3.1-8b-instant",
        api_base="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
        is_chat_model=True,
        context_window=131072,
        max_tokens=512,
    )
else:
    from llama_index.core.embeddings import MockEmbedding
    from llama_index.core.llms import MockLLM
    Settings.embed_model = MockEmbedding(embed_dim=384)
    Settings.llm = MockLLM()

print(f"Embedding model: {type(Settings.embed_model).__name__}")
print(f"LLM: {type(Settings.llm).__name__}\n")

# ---------------------------------------------------------------------------
# STEP 1: Load CSV
# ---------------------------------------------------------------------------
def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns found: {list(df.columns)}\n")

    missing = [expected for expected, actual in COLUMN_MAP.items() if actual not in df.columns]
    if missing:
        print(f"WARNING: these expected fields are missing from the CSV: {missing}")
        print("Check COLUMN_MAP at the top of this script against the columns printed above.\n")

    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        print(f"Sampled down to {len(df)} rows for this run.\n")

    # Drop rows with no restaurant name — can't build a meaningful document
    name_col = COLUMN_MAP["name"]
    before = len(df)
    df = df.dropna(subset=[name_col]).reset_index(drop=True)
    if before != len(df):
        print(f"Dropped {before - len(df)} rows with missing name.\n")

    return df


def col(row: pd.Series, key: str, default=""):
    """Fetch a column via COLUMN_MAP, tolerating missing columns/values."""
    actual_col = COLUMN_MAP.get(key)
    if actual_col is None or actual_col not in row.index:
        return default
    val = row[actual_col]
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return val


def sanitize_metadata(meta: dict) -> dict:
    """Chroma/JSON-safe metadata: replace NaN/NaT with None."""
    clean = {}
    for k, v in meta.items():
        if isinstance(v, float) and math.isnan(v):
            clean[k] = None
        else:
            clean[k] = v
    return clean


# ---------------------------------------------------------------------------
# STEP 2: Parsing utilities
# ---------------------------------------------------------------------------
def parse_reviews(raw_reviews_list) -> str:
    if not raw_reviews_list or not isinstance(raw_reviews_list, str):
        return ""
    try:
        parsed = literal_eval(raw_reviews_list)
        texts = [str(r[1]).strip() for r in parsed if isinstance(r, (tuple, list)) and len(r) >= 2]
        return " ".join(texts)
    except (ValueError, SyntaxError):
        return ""


def parse_menu_items(raw_menu) -> list:
    if not raw_menu or not isinstance(raw_menu, str):
        return []
    try:
        parsed = literal_eval(raw_menu)
        return [str(m).strip() for m in parsed] if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


# ---------------------------------------------------------------------------
# STEP 3: Document builder
# ---------------------------------------------------------------------------
def build_restaurant_document(row: pd.Series) -> Document:
    name = col(row, "name")
    rate_raw = col(row, "rate", default=None)
    rate_is_imputed = False  # TODO: if your imputation pipeline tracks this in a
                              # dedicated column (e.g. `rate_was_null`), read it here
                              # instead of hardcoding False.

    review_text = parse_reviews(col(row, "reviews_list"))
    menu_items = parse_menu_items(col(row, "menu_item"))
    menu_available = len(menu_items) > 0
    menu_clause = f"Menu highlights: {', '.join(menu_items)}." if menu_available else ""

    text_content = (
        f"{name} is a {col(row, 'rest_type', 'restaurant')} located in {col(row, 'location', 'an unspecified area')}, "
        f"serving {col(row, 'cuisines', 'unspecified cuisine')}. "
        f"Popular dishes: {col(row, 'dish_liked') or 'not specified'}. "
        f"Approximate cost for two people: Rs.{col(row, 'cost_for_two', 'unspecified')}. "
        f"{menu_clause} "
        f"Customer reviews: {review_text or 'No reviews available.'}"
    )

    metadata = sanitize_metadata({
        "doc_type": "restaurant",
        "name": name,
        "location": col(row, "location", default=None),
        "cuisines": col(row, "cuisines", default=None),
        "rest_type": col(row, "rest_type", default=None),
        "cost_for_two": col(row, "cost_for_two", default=None),
        "rating": rate_raw,
        "rating_is_imputed": rate_is_imputed,
        "online_order": col(row, "online_order", default=None),
        "book_table": col(row, "book_table", default=None),
        "menu_available": menu_available,
    })
    return Document(text=text_content, metadata=metadata)


# Optional FAQ documents — replace with your own scraped Q&A pairs.
# You have web_fetch access on your machine to pull Zomato's help pages;
# summarize each into a Q/A pair here rather than copy-pasting site text verbatim.
DEFAULT_FAQS = [
    {"question": "How do I cancel my Zomato order?",
     "answer": "Go to Orders, select the active order, and tap Cancel. Refunds are processed within 5-7 business days."},
    {"question": "How does Zomato table booking work?",
     "answer": "Search a restaurant, check slot availability, and confirm. Some restaurants require a refundable deposit."},
]


def build_faq_document(faq: dict) -> Document:
    return Document(text=f"Q: {faq['question']} A: {faq['answer']}",
                     metadata={"doc_type": "faq", "question": faq["question"]})


# ---------------------------------------------------------------------------
# STEP 4: Build the index (idempotent)
# ---------------------------------------------------------------------------
def build_index() -> VectorStoreIndex:
    from llama_index.vector_stores.chroma import ChromaVectorStore
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    # --- Fast path: reuse existing embeddings, skip CSV load + re-embedding entirely ---
    if not REBUILD_INDEX:
        try:
            chroma_collection = chroma_client.get_collection(COLLECTION_NAME)
            existing_count = chroma_collection.count()
        except Exception:
            existing_count = 0

        if existing_count > 0:
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            index = VectorStoreIndex.from_vector_store(vector_store)
            print(f"Loaded existing index with {existing_count} documents from {CHROMA_PATH} "
                  f"— NO re-embedding performed.\n")
            return index
        else:
            print("REBUILD_INDEX is False but no existing collection was found — building fresh.\n")

    # --- Full rebuild path: only reached if REBUILD_INDEX=True or nothing exists yet ---
    df = load_dataframe(CSV_PATH)
    restaurant_docs = [build_restaurant_document(row) for _, row in df.iterrows()]
    faq_docs = [build_faq_document(f) for f in DEFAULT_FAQS]
    all_documents = restaurant_docs + faq_docs
    print(f"Built {len(restaurant_docs)} restaurant docs + {len(faq_docs)} FAQ docs = {len(all_documents)} total.\n")

    if REBUILD_INDEX:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
            print(f"Cleared existing '{COLLECTION_NAME}' collection before rebuild.\n")
        except Exception:
            pass  # collection didn't exist yet — fine

    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        all_documents, storage_context=storage_context, show_progress=True
    )
    print(f"\nIndexed {chroma_collection.count()} documents into ChromaDB at {CHROMA_PATH}\n")
    return index


# ---------------------------------------------------------------------------
# STEP 5: Interactive chat loop
# ---------------------------------------------------------------------------
def run_chat(index: VectorStoreIndex):
    chat_engine = index.as_chat_engine(chat_mode="condense_plus_context", similarity_top_k=3)
    print("Zomato RAG chatbot ready. Type 'exit' to quit, 'sources' to see last retrieval.\n")
    last_response = None
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in {"exit", "quit"}:
            break
        if query.lower() == "sources" and last_response is not None:
            for n in last_response.source_nodes:
                print(f"  - [{n.metadata.get('doc_type')}] {n.metadata.get('name', n.metadata.get('question'))} (score={n.score:.3f})")
            continue
        last_response = chat_engine.chat(query)
        print(f"Bot: {last_response}\n")


if __name__ == "__main__":
    idx = build_index()
    if "--no-chat" not in sys.argv:
        run_chat(idx)
